// Package scan streams tensor bytes in fixed-size chunks and folds them
// into SHA-256 digests, numeric statistics, and — for tensor pairs read
// in lockstep — elementwise difference metrics. Memory use is constant
// (one or two chunk buffers) regardless of tensor or checkpoint size.
package scan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/dtype"
)

// chunkSize is the per-side read buffer: 256 KiB, a multiple of every
// element size layerdiff knows, so no element ever straddles two chunks.
const chunkSize = 256 << 10

// F64 is a float64 that survives JSON encoding when non-finite:
// encoding/json rejects NaN and ±Inf, so those render as strings.
type F64 float64

// MarshalJSON encodes finite values as numbers and non-finite values as
// the strings "NaN", "Infinity", and "-Infinity".
func (f F64) MarshalJSON() ([]byte, error) {
	v := float64(f)
	switch {
	case math.IsNaN(v):
		return []byte(`"NaN"`), nil
	case math.IsInf(v, 1):
		return []byte(`"Infinity"`), nil
	case math.IsInf(v, -1):
		return []byte(`"-Infinity"`), nil
	}
	return json.Marshal(v)
}

// Stats summarizes one tensor's values in a single streamed pass.
// Min/Max/Mean/RMS are computed over non-NaN elements (±Inf included,
// which can make Mean and RMS infinite); they are NaN when the tensor
// has no non-NaN element. L2 is the Euclidean norm over the same set.
type Stats struct {
	Count int64 `json:"count"`
	Zeros int64 `json:"zeros"`
	NaNs  int64 `json:"nans"`
	Infs  int64 `json:"infs"`
	Min   F64   `json:"min"`
	Max   F64   `json:"max"`
	Mean  F64   `json:"mean"`
	RMS   F64   `json:"rms"`
	L2    F64   `json:"l2"`
}

// Result is the outcome of scanning one tensor.
type Result struct {
	// SHA256 is the hex digest of the raw little-endian tensor bytes.
	SHA256 string
	Bytes  int64
	// Stats is nil for dtypes layerdiff cannot decode numerically.
	Stats *Stats
}

// statsAcc folds decoded elements into running statistics.
type statsAcc struct {
	count, zeros, nans, infs int64
	min, max                 float64
	sum, sumSq               float64
	seen                     bool // at least one non-NaN element
}

func (s *statsAcc) add(v float64) {
	s.count++
	if math.IsNaN(v) {
		s.nans++
		return
	}
	if math.IsInf(v, 0) {
		s.infs++
	}
	if v == 0 {
		s.zeros++
	}
	if !s.seen {
		s.min, s.max = v, v
		s.seen = true
	} else {
		if v < s.min {
			s.min = v
		}
		if v > s.max {
			s.max = v
		}
	}
	s.sum += v
	s.sumSq += v * v
}

func (s *statsAcc) stats() *Stats {
	st := &Stats{Count: s.count, Zeros: s.zeros, NaNs: s.nans, Infs: s.infs}
	if !s.seen {
		nan := F64(math.NaN())
		st.Min, st.Max, st.Mean, st.RMS = nan, nan, nan, nan
		return st
	}
	n := float64(s.count - s.nans)
	st.Min = F64(s.min)
	st.Max = F64(s.max)
	st.Mean = F64(s.sum / n)
	st.RMS = F64(math.Sqrt(s.sumSq / n))
	st.L2 = F64(math.Sqrt(s.sumSq))
	return st
}

// Tensor streams one tensor once, computing its digest and — when the
// dtype is numerically decodable — its statistics.
func Tensor(t *checkpoint.Tensor) (Result, error) {
	info := dtype.Lookup(t.DType)
	var acc *statsAcc
	if info.Numeric() {
		acc = &statsAcc{}
	}

	r, err := t.Reader()
	if err != nil {
		return Result{}, err
	}
	defer r.Close()

	h := sha256.New()
	buf := make([]byte, chunkSize)
	remaining := t.Bytes
	for remaining > 0 {
		n := int64(len(buf))
		if remaining < n {
			n = remaining
		}
		if _, err := io.ReadFull(r, buf[:n]); err != nil {
			return Result{}, fmt.Errorf("tensor %q: read failed: %w", t.Name, err)
		}
		h.Write(buf[:n])
		if acc != nil {
			size := int64(info.Size)
			for i := int64(0); i < n; i += size {
				acc.add(info.Decode(buf[i : i+size]))
			}
		}
		remaining -= n
	}

	res := Result{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: t.Bytes}
	if acc != nil {
		res.Stats = acc.stats()
	}
	return res, nil
}

// HashOnly streams one tensor computing only its digest — the fast path
// used by manifest building and --hash-only diffs.
func HashOnly(t *checkpoint.Tensor) (Result, error) {
	r, err := t.Reader()
	if err != nil {
		return Result{}, err
	}
	defer r.Close()

	h := sha256.New()
	n, err := io.CopyBuffer(h, r, make([]byte, chunkSize))
	if err != nil {
		return Result{}, fmt.Errorf("tensor %q: read failed: %w", t.Name, err)
	}
	if n != t.Bytes {
		// The shard shrank between header parse and read (or is truncated).
		return Result{}, fmt.Errorf("tensor %q: expected %d bytes, read %d", t.Name, t.Bytes, n)
	}
	return Result{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: t.Bytes}, nil
}
