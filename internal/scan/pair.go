// Lockstep pair scanning: both tensors stream chunk by chunk through two
// fixed buffers, producing per-side digests and statistics plus
// elementwise difference metrics in a single pass over each file.
package scan

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/dtype"
)

// Tolerance decides when an element pair counts as changed:
// |b−a| > Atol + Rtol×|b|, following the numpy.isclose convention with
// the second (new) checkpoint as the reference. The zero value demands
// exact numeric equality.
type Tolerance struct {
	Atol float64
	Rtol float64
}

// DiffMetrics summarizes the elementwise difference b−a of two tensors
// with identical dtype and shape.
//
// NaN handling: a pair of NaNs is treated as equal (it neither counts as
// changed nor contributes to the norms); a NaN facing a number counts as
// changed but is excluded from the numeric aggregates. A pair of
// same-signed infinities is treated as equal.
type DiffMetrics struct {
	// Elems is the total element count; NumericPairs the pairs where both
	// sides were non-NaN and therefore entered the aggregates below.
	Elems        int64 `json:"elems"`
	NumericPairs int64 `json:"numeric_pairs"`
	// Changed counts elements outside the tolerance, plus NaN mismatches.
	Changed int64 `json:"changed"`
	// MaxAbs is the largest |b−a|; MaxAbsIndex its flat element index.
	MaxAbs      F64   `json:"max_abs"`
	MaxAbsIndex int64 `json:"max_abs_index"`
	// MeanAbs is Σ|b−a| / NumericPairs; L1 and L2 the difference norms.
	MeanAbs F64 `json:"mean_abs"`
	L1      F64 `json:"l1"`
	L2      F64 `json:"l2"`
	// RelL2 is ‖b−a‖₂ / ‖a‖₂ (0 when both are zero, +Inf when only the
	// base norm is zero).
	RelL2 F64 `json:"rel_l2"`
	// Cosine is the cosine similarity of the two tensors seen as flat
	// vectors; NaN when either norm is zero.
	Cosine F64 `json:"cosine"`
}

// PairResult carries both per-side scan results, byte-level equality,
// and — when the dtype is numerically decodable — difference metrics.
type PairResult struct {
	A, B         Result
	BitwiseEqual bool
	// Metrics is nil for known-size but non-decodable dtypes (e.g. FP8),
	// where only byte-level comparison is possible.
	Metrics *DiffMetrics
}

// pairAcc folds elementwise differences.
type pairAcc struct {
	elems, numeric, changed int64
	maxAbs                  float64
	maxIdx                  int64
	sumAbs, sumSq, dot      float64
}

func (p *pairAcc) add(va, vb float64, tol Tolerance) {
	idx := p.elems
	p.elems++
	aNaN, bNaN := math.IsNaN(va), math.IsNaN(vb)
	if aNaN || bNaN {
		if !(aNaN && bNaN) {
			p.changed++
		}
		return
	}
	p.numeric++
	p.dot += va * vb
	d := vb - va
	if math.IsInf(va, 0) && math.IsInf(vb, 0) && (va > 0) == (vb > 0) {
		d = 0 // same-signed infinities: Inf−Inf is NaN, but the pair agrees
	}
	ad := math.Abs(d)
	p.sumAbs += ad
	p.sumSq += d * d
	if ad > p.maxAbs {
		p.maxAbs = ad
		p.maxIdx = idx
	}
	if ad > tol.Atol+tol.Rtol*math.Abs(vb) {
		p.changed++
	}
}

func (p *pairAcc) metrics(baseL2 float64) *DiffMetrics {
	m := &DiffMetrics{
		Elems:        p.elems,
		NumericPairs: p.numeric,
		Changed:      p.changed,
		MaxAbs:       F64(p.maxAbs),
		MaxAbsIndex:  p.maxIdx,
		L1:           F64(p.sumAbs),
		L2:           F64(math.Sqrt(p.sumSq)),
	}
	if p.numeric > 0 {
		m.MeanAbs = F64(p.sumAbs / float64(p.numeric))
	} else {
		m.MeanAbs = F64(math.NaN())
	}
	l2 := math.Sqrt(p.sumSq)
	switch {
	case baseL2 > 0:
		m.RelL2 = F64(l2 / baseL2)
	case l2 == 0:
		m.RelL2 = 0
	default:
		m.RelL2 = F64(math.Inf(1))
	}
	// Cosine needs both norms; the caller passes ‖a‖₂, ‖b‖₂ separately.
	return m
}

// Pair streams tensors a and b in lockstep. Both must have the same
// dtype and byte length — the caller guarantees that by classifying
// dtype/shape mismatches before calling.
func Pair(a, b *checkpoint.Tensor, tol Tolerance) (PairResult, error) {
	if a.DType != b.DType || a.Bytes != b.Bytes {
		return PairResult{}, fmt.Errorf("pair scan requires matching dtype and size: %s/%d vs %s/%d",
			a.DType, a.Bytes, b.DType, b.Bytes)
	}
	info := dtype.Lookup(a.DType)

	ra, err := a.Reader()
	if err != nil {
		return PairResult{}, err
	}
	defer ra.Close()
	rb, err := b.Reader()
	if err != nil {
		return PairResult{}, err
	}
	defer rb.Close()

	ha, hb := sha256.New(), sha256.New()
	bufA := make([]byte, chunkSize)
	bufB := make([]byte, chunkSize)
	var accA, accB *statsAcc
	var diff *pairAcc
	if info.Numeric() {
		accA, accB = &statsAcc{}, &statsAcc{}
		diff = &pairAcc{}
	}

	bitwise := true
	remaining := a.Bytes
	for remaining > 0 {
		n := int64(len(bufA))
		if remaining < n {
			n = remaining
		}
		if _, err := io.ReadFull(ra, bufA[:n]); err != nil {
			return PairResult{}, fmt.Errorf("tensor %q: read failed: %w", a.Name, err)
		}
		if _, err := io.ReadFull(rb, bufB[:n]); err != nil {
			return PairResult{}, fmt.Errorf("tensor %q: read failed: %w", b.Name, err)
		}
		ha.Write(bufA[:n])
		hb.Write(bufB[:n])
		if bitwise && !bytes.Equal(bufA[:n], bufB[:n]) {
			bitwise = false
		}
		if diff != nil {
			size := int64(info.Size)
			for i := int64(0); i < n; i += size {
				va := info.Decode(bufA[i : i+size])
				vb := info.Decode(bufB[i : i+size])
				accA.add(va)
				accB.add(vb)
				diff.add(va, vb, tol)
			}
		}
		remaining -= n
	}

	res := PairResult{
		A:            Result{SHA256: hex.EncodeToString(ha.Sum(nil)), Bytes: a.Bytes},
		B:            Result{SHA256: hex.EncodeToString(hb.Sum(nil)), Bytes: b.Bytes},
		BitwiseEqual: bitwise,
	}
	if diff != nil {
		res.A.Stats = accA.stats()
		res.B.Stats = accB.stats()
		m := diff.metrics(float64(res.A.Stats.L2))
		m.Cosine = cosine(diff.dot, float64(res.A.Stats.L2), float64(res.B.Stats.L2))
		res.Metrics = m
	}
	return res, nil
}

// cosine returns dot/(‖a‖·‖b‖), or NaN when either norm is zero.
func cosine(dot, na, nb float64) F64 {
	if na == 0 || nb == 0 {
		return F64(math.NaN())
	}
	return F64(dot / (na * nb))
}
