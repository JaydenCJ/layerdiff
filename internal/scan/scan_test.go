// Tests for single-tensor scanning: digests must match an independent
// SHA-256 of the raw bytes, and the streamed statistics must equal what
// a whole-tensor computation would give.
package scan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// open builds a one-off checkpoint in a temp dir and returns the named
// tensor, ready to scan.
func open(t *testing.T, tensors []stwrite.Tensor, name string) *checkpoint.Tensor {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.safetensors")
	if err := stwrite.WriteFile(path, nil, tensors); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	cp, err := checkpoint.Open(path)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	tt, ok := cp.Tensor(name)
	if !ok {
		t.Fatalf("tensor %q missing from fixture", name)
	}
	return tt
}

func TestTensorHashMatchesIndependentSHA256(t *testing.T) {
	raw := stwrite.F32(0.5, -1.25, 3)
	tt := open(t, []stwrite.Tensor{{Name: "w", DType: "F32", Shape: []int64{3}, Data: raw}}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	sum := sha256.Sum256(raw)
	if res.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("digest mismatch: got %s", res.SHA256)
	}
	if res.Bytes != 12 {
		t.Fatalf("bytes = %d, want 12", res.Bytes)
	}

	// The hash-only fast path must agree with the full scan.
	fast, err := HashOnly(tt)
	if err != nil {
		t.Fatal(err)
	}
	if fast.SHA256 != res.SHA256 || fast.Bytes != res.Bytes {
		t.Fatalf("fast path diverged: %s vs %s", fast.SHA256, res.SHA256)
	}
	if fast.Stats != nil {
		t.Fatal("HashOnly must not compute stats")
	}
}

func TestTensorStatsKnownValues(t *testing.T) {
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F32", Shape: []int64{4}, Data: stwrite.F32(1, -2, 0, 3)},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	s := res.Stats
	if s == nil {
		t.Fatal("F32 tensor must have stats")
	}
	if s.Count != 4 || s.Zeros != 1 || s.NaNs != 0 || s.Infs != 0 {
		t.Fatalf("counts wrong: %+v", s)
	}
	if float64(s.Min) != -2 || float64(s.Max) != 3 || float64(s.Mean) != 0.5 {
		t.Fatalf("min/max/mean wrong: %+v", s)
	}
	wantRMS := math.Sqrt(14.0 / 4.0)
	if math.Abs(float64(s.RMS)-wantRMS) > 1e-12 {
		t.Fatalf("rms = %v, want %v", s.RMS, wantRMS)
	}
	if math.Abs(float64(s.L2)-math.Sqrt(14)) > 1e-12 {
		t.Fatalf("l2 = %v, want sqrt(14)", s.L2)
	}
}

func TestTensorStatsCountNaNAndInf(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F32", Shape: []int64{4}, Data: stwrite.F32(nan, inf, -1, 0)},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	s := res.Stats
	if s.NaNs != 1 || s.Infs != 1 || s.Zeros != 1 {
		t.Fatalf("special counts wrong: %+v", s)
	}
	// NaN is excluded from min/max; Inf is not.
	if float64(s.Min) != -1 || !math.IsInf(float64(s.Max), 1) {
		t.Fatalf("min/max with specials wrong: min=%v max=%v", s.Min, s.Max)
	}
}

func TestTensorAllNaNStatsAreNaN(t *testing.T) {
	nan := float32(math.NaN())
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(nan, nan)},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	s := res.Stats
	if !math.IsNaN(float64(s.Min)) || !math.IsNaN(float64(s.Mean)) {
		t.Fatalf("all-NaN tensor must have NaN min/mean: %+v", s)
	}
	if s.NaNs != 2 || s.Count != 2 {
		t.Fatalf("counts wrong: %+v", s)
	}
}

func TestTensorEmpty(t *testing.T) {
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F32", Shape: []int64{0, 8}, Data: nil},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	sum := sha256.Sum256(nil)
	if res.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatal("empty tensor must hash to SHA-256 of empty input")
	}
	if res.Stats.Count != 0 {
		t.Fatalf("empty tensor count = %d", res.Stats.Count)
	}
}

func TestTensorOpaqueDTypeHashesWithoutStats(t *testing.T) {
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F8_E4M3", Shape: []int64{3}, Data: []byte{1, 2, 3}},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if res.Stats != nil {
		t.Fatal("non-numeric dtype must not produce stats")
	}
	if res.SHA256 == "" {
		t.Fatal("non-numeric dtype must still hash")
	}
}

func TestF16StatsDecodeCorrectly(t *testing.T) {
	// 0x3c00 = 1.0, 0xc000 = -2.0 in binary16.
	tt := open(t, []stwrite.Tensor{
		{Name: "w", DType: "F16", Shape: []int64{2}, Data: stwrite.U16(0x3c00, 0xc000)},
	}, "w")
	res, err := Tensor(tt)
	if err != nil {
		t.Fatal(err)
	}
	if float64(res.Stats.Min) != -2 || float64(res.Stats.Max) != 1 || float64(res.Stats.Mean) != -0.5 {
		t.Fatalf("F16 stats wrong: %+v", res.Stats)
	}
}

func TestF64JSONMarshalNonFinite(t *testing.T) {
	// encoding/json rejects NaN/Inf float64; F64 must not.
	in := struct {
		A F64 `json:"a"`
		B F64 `json:"b"`
		C F64 `json:"c"`
		D F64 `json:"d"`
	}{F64(math.NaN()), F64(math.Inf(1)), F64(math.Inf(-1)), F64(1.5)}
	out, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"a":"NaN","b":"Infinity","c":"-Infinity","d":1.5}`
	if string(out) != want {
		t.Fatalf("got %s, want %s", out, want)
	}
}
