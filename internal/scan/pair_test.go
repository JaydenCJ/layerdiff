// Tests for lockstep pair scanning: the difference metrics are checked
// against hand-computed values, and the NaN/Inf/negative-zero edge cases
// are pinned down because they decide whether a tensor counts as
// changed.
package scan

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// openPair writes two single-tensor files and returns both tensors.
func openPair(t *testing.T, a, b stwrite.Tensor) (*checkpoint.Tensor, *checkpoint.Tensor) {
	t.Helper()
	dir := t.TempDir()
	pa := filepath.Join(dir, "a.safetensors")
	pb := filepath.Join(dir, "b.safetensors")
	if err := stwrite.WriteFile(pa, nil, []stwrite.Tensor{a}); err != nil {
		t.Fatal(err)
	}
	if err := stwrite.WriteFile(pb, nil, []stwrite.Tensor{b}); err != nil {
		t.Fatal(err)
	}
	ca, err := checkpoint.Open(pa)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := checkpoint.Open(pb)
	if err != nil {
		t.Fatal(err)
	}
	ta, _ := ca.Tensor(a.Name)
	tb, _ := cb.Tensor(b.Name)
	return ta, tb
}

func f32Tensor(name string, vals ...float32) stwrite.Tensor {
	return stwrite.Tensor{Name: name, DType: "F32", Shape: []int64{int64(len(vals))}, Data: stwrite.F32(vals...)}
}

func TestPairIdenticalTensorsAreBitwiseEqual(t *testing.T) {
	ta, tb := openPair(t, f32Tensor("w", 1, 2, 3), f32Tensor("w", 1, 2, 3))
	pr, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if !pr.BitwiseEqual || pr.A.SHA256 != pr.B.SHA256 {
		t.Fatalf("identical tensors must be bitwise equal: %+v", pr)
	}
	if pr.Metrics.Changed != 0 || float64(pr.Metrics.MaxAbs) != 0 {
		t.Fatalf("identical tensors must have zero diff: %+v", pr.Metrics)
	}
}

func TestPairMetricsHandComputed(t *testing.T) {
	// a = [1,2,3,4], b = [1,2,4,6]: diffs are [0,0,1,2].
	ta, tb := openPair(t, f32Tensor("w", 1, 2, 3, 4), f32Tensor("w", 1, 2, 4, 6))
	pr, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	m := pr.Metrics
	if pr.BitwiseEqual {
		t.Fatal("differing tensors reported bitwise equal")
	}
	if m.Elems != 4 || m.NumericPairs != 4 || m.Changed != 2 {
		t.Fatalf("counts wrong: %+v", m)
	}
	if float64(m.MaxAbs) != 2 || m.MaxAbsIndex != 3 {
		t.Fatalf("max diff wrong: %+v", m)
	}
	if float64(m.L1) != 3 || float64(m.MeanAbs) != 0.75 {
		t.Fatalf("L1/mean wrong: %+v", m)
	}
	if math.Abs(float64(m.L2)-math.Sqrt(5)) > 1e-12 {
		t.Fatalf("L2 = %v, want sqrt(5)", m.L2)
	}
	wantRel := math.Sqrt(5) / math.Sqrt(30) // ‖diff‖ / ‖a‖
	if math.Abs(float64(m.RelL2)-wantRel) > 1e-12 {
		t.Fatalf("RelL2 = %v, want %v", m.RelL2, wantRel)
	}
	wantCos := (1.0 + 4 + 12 + 24) / (math.Sqrt(30) * math.Sqrt(57))
	if math.Abs(float64(m.Cosine)-wantCos) > 1e-9 {
		t.Fatalf("cosine = %v, want %v", m.Cosine, wantCos)
	}
}

func TestPairAtolSuppressesSmallChanges(t *testing.T) {
	ta, tb := openPair(t, f32Tensor("w", 1, 2), f32Tensor("w", 1.0005, 2))
	strict, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if strict.Metrics.Changed != 1 {
		t.Fatalf("without tolerance the element must count as changed: %+v", strict.Metrics)
	}
	loose, err := Pair(ta, tb, Tolerance{Atol: 1e-3})
	if err != nil {
		t.Fatal(err)
	}
	if loose.Metrics.Changed != 0 {
		t.Fatalf("atol=1e-3 must absorb a 5e-4 delta: %+v", loose.Metrics)
	}
	if loose.BitwiseEqual {
		t.Fatal("tolerance must not affect bitwise equality")
	}
}

func TestPairRtolScalesWithMagnitude(t *testing.T) {
	// Same absolute delta (0.5) on elements of very different size:
	// 1000.5 vs 1000 is within rtol 1e-3, 1.5 vs 1 is not.
	ta, tb := openPair(t, f32Tensor("w", 1000, 1), f32Tensor("w", 1000.5, 1.5))
	pr, err := Pair(ta, tb, Tolerance{Rtol: 1e-3})
	if err != nil {
		t.Fatal(err)
	}
	if pr.Metrics.Changed != 1 {
		t.Fatalf("rtol must pass the large element and flag the small one: %+v", pr.Metrics)
	}
}

func TestPairNaNPayloadsAreEquivalentNotIdentical(t *testing.T) {
	// Two different NaN bit patterns: bitwise different, numerically
	// "both NaN" — the pair must not count as changed.
	a := stwrite.Tensor{Name: "w", DType: "F32", Shape: []int64{1},
		Data: []byte{0x01, 0x00, 0xc0, 0x7f}} // NaN payload 1
	b := stwrite.Tensor{Name: "w", DType: "F32", Shape: []int64{1},
		Data: []byte{0x02, 0x00, 0xc0, 0x7f}} // NaN payload 2
	ta, tb := openPair(t, a, b)
	pr, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if pr.BitwiseEqual {
		t.Fatal("distinct payloads must not be bitwise equal")
	}
	if pr.Metrics.Changed != 0 {
		t.Fatalf("NaN vs NaN must not count as changed: %+v", pr.Metrics)
	}
}

func TestPairNaNVersusNumberCountsChanged(t *testing.T) {
	ta, tb := openPair(t, f32Tensor("w", float32(math.NaN()), 2), f32Tensor("w", 1, 2))
	pr, err := Pair(ta, tb, Tolerance{Atol: 100}) // even a huge tolerance can't excuse NaN
	if err != nil {
		t.Fatal(err)
	}
	if pr.Metrics.Changed != 1 || pr.Metrics.NumericPairs != 1 {
		t.Fatalf("NaN vs number must count as changed: %+v", pr.Metrics)
	}
}

func TestPairSpecialValueEquivalences(t *testing.T) {
	// Inf−Inf is NaN in IEEE math; the scanner must special-case
	// same-signed infinities as agreeing.
	inf := float32(math.Inf(1))
	ta, tb := openPair(t, f32Tensor("w", inf, 1), f32Tensor("w", inf, 1))
	pr, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if pr.Metrics.Changed != 0 || float64(pr.Metrics.MaxAbs) != 0 {
		t.Fatalf("equal infinities must not count as changed: %+v", pr.Metrics)
	}

	// 0.0 vs -0.0: different bytes, numerically equal.
	negZero := float32(math.Copysign(0, -1))
	ta, tb = openPair(t, f32Tensor("w", 0), f32Tensor("w", negZero))
	pr, err = Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if pr.BitwiseEqual {
		t.Fatal("0.0 and -0.0 differ bitwise")
	}
	if pr.Metrics.Changed != 0 {
		t.Fatalf("0.0 vs -0.0 must not count as changed: %+v", pr.Metrics)
	}
}

func TestPairOpaqueDTypeComparesBytesOnly(t *testing.T) {
	a := stwrite.Tensor{Name: "w", DType: "F8_E4M3", Shape: []int64{2}, Data: []byte{1, 2}}
	b := stwrite.Tensor{Name: "w", DType: "F8_E4M3", Shape: []int64{2}, Data: []byte{1, 3}}
	ta, tb := openPair(t, a, b)
	pr, err := Pair(ta, tb, Tolerance{})
	if err != nil {
		t.Fatal(err)
	}
	if pr.Metrics != nil {
		t.Fatal("opaque dtype must not produce metrics")
	}
	if pr.BitwiseEqual {
		t.Fatal("differing bytes must not be bitwise equal")
	}
}

func TestPairRejectsMismatchedInput(t *testing.T) {
	ta, tb := openPair(t,
		f32Tensor("w", 1, 2),
		stwrite.Tensor{Name: "w", DType: "F64", Shape: []int64{2}, Data: stwrite.F64(1, 2)})
	if _, err := Pair(ta, tb, Tolerance{}); err == nil {
		t.Fatal("mismatched dtype must be rejected by Pair")
	}
}
