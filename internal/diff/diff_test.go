// Tests for the diff orchestrator: classification of every tensor fate,
// layer rollups, tolerance behavior, filtering, and the determinism
// guarantee that identical inputs yield identical reports.
package diff

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/manifest"
	"github.com/JaydenCJ/layerdiff/internal/scan"
	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// cp writes tensors into a fresh checkpoint file and opens it.
func cp(t *testing.T, tensors ...stwrite.Tensor) *checkpoint.Checkpoint {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := stwrite.WriteFile(path, nil, tensors); err != nil {
		t.Fatal(err)
	}
	c, err := checkpoint.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func f32(name string, vals ...float32) stwrite.Tensor {
	return stwrite.Tensor{Name: name, DType: "F32", Shape: []int64{int64(len(vals))}, Data: stwrite.F32(vals...)}
}

func classes(r *Report) map[string]Class {
	out := map[string]Class{}
	for _, td := range r.Tensors {
		out[td.Name] = td.Class
	}
	return out
}

func TestIdenticalCheckpoints(t *testing.T) {
	mk := func() *checkpoint.Checkpoint {
		return cp(t, f32("model.layers.0.w", 1, 2), f32("model.norm.weight", 3))
	}
	rep, err := Checkpoints(mk(), mk(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Different() {
		t.Fatal("identical checkpoints reported different")
	}
	s := rep.Summary
	if s.Identical != 2 || s.Compared != 2 || len(rep.Tensors) != 0 || len(rep.Layers) != 0 {
		t.Fatalf("summary wrong: %+v tensors=%d layers=%d", s, len(rep.Tensors), len(rep.Layers))
	}
	if s.Layers != 2 || s.LayersChanged != 0 {
		t.Fatalf("layer counts wrong: %+v", s)
	}
}

func TestChangedAddedRemovedClassification(t *testing.T) {
	a := cp(t,
		f32("model.layers.0.w", 1, 2),
		f32("model.layers.1.w", 3, 4),
		f32("only.in.a", 5))
	b := cp(t,
		f32("model.layers.0.w", 1, 2),
		f32("model.layers.1.w", 3, 9), // changed
		f32("only.in.b", 6))

	rep, err := Checkpoints(a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := classes(rep)
	want := map[string]Class{
		"model.layers.1.w": ClassChanged,
		"only.in.a":        ClassRemoved,
		"only.in.b":        ClassAdded,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("classes = %v, want %v", got, want)
	}
	if !rep.Different() {
		t.Fatal("must be different")
	}
	s := rep.Summary
	if s.Identical != 1 || s.Changed != 1 || s.Added != 1 || s.Removed != 1 {
		t.Fatalf("summary wrong: %+v", s)
	}
	// One-sided tensors still carry hash and stats from their side.
	for _, td := range rep.Tensors {
		if td.Class == ClassRemoved && (td.SHA256A == "" || td.StatsA == nil) {
			t.Fatalf("removed tensor missing side-A scan: %+v", td)
		}
		if td.Class == ClassAdded && (td.SHA256B == "" || td.StatsB == nil) {
			t.Fatalf("added tensor missing side-B scan: %+v", td)
		}
	}
}

func TestDTypeAndShapeMismatch(t *testing.T) {
	a := cp(t,
		f32("w.dtype", 1, 2),
		f32("w.shape", 1, 2, 3, 4))
	b := cp(t,
		stwrite.Tensor{Name: "w.dtype", DType: "F64", Shape: []int64{2}, Data: stwrite.F64(1, 2)},
		stwrite.Tensor{Name: "w.shape", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(1, 2, 3, 4)})

	rep, err := Checkpoints(a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := classes(rep)
	if got["w.dtype"] != ClassDTypeMismatch || got["w.shape"] != ClassShapeMismatch {
		t.Fatalf("classes = %v", got)
	}
	if rep.Summary.Mismatched != 2 {
		t.Fatalf("mismatched = %d", rep.Summary.Mismatched)
	}
}

func TestToleranceMakesTensorEquivalent(t *testing.T) {
	a := cp(t, f32("w", 1, 2))
	b := cp(t, f32("w", 1.0000001, 2))
	strict, err := Checkpoints(a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if classes(strict)["w"] != ClassChanged || !strict.Different() {
		t.Fatalf("without tolerance: %v", classes(strict))
	}
	loose, err := Checkpoints(a, b, Options{Tolerance: scan.Tolerance{Atol: 1e-6}})
	if err != nil {
		t.Fatal(err)
	}
	if classes(loose)["w"] != ClassEquivalent {
		t.Fatalf("with tolerance: %v", classes(loose))
	}
	if loose.Different() {
		t.Fatal("equivalent-only diff must not count as different")
	}
	if loose.Summary.Equivalent != 1 {
		t.Fatalf("summary: %+v", loose.Summary)
	}
}

func TestLayerRollup(t *testing.T) {
	a := cp(t,
		f32("model.layers.0.attn.w", 1, 2),
		f32("model.layers.0.mlp.w", 3, 4),
		f32("model.layers.1.attn.w", 5, 6),
		f32("model.embed.weight", 7))
	b := cp(t,
		f32("model.layers.0.attn.w", 1, 4), // Δ=[0,2]
		f32("model.layers.0.mlp.w", 3, 5),  // Δ=[0,1]
		f32("model.layers.1.attn.w", 5, 6), // identical
		f32("model.embed.weight", 7))       // identical

	rep, err := Checkpoints(a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Summary.Layers != 3 || rep.Summary.LayersChanged != 1 {
		t.Fatalf("layer counts: %+v", rep.Summary)
	}
	if len(rep.Layers) != 1 || rep.Layers[0].Layer != "model.layers.0" {
		t.Fatalf("layer rows: %+v", rep.Layers)
	}
	row := rep.Layers[0]
	if row.Tensors != 2 || row.Changed != 2 {
		t.Fatalf("row counts: %+v", row)
	}
	if float64(row.MaxAbs) != 2 {
		t.Fatalf("layer max|Δ| = %v, want 2", row.MaxAbs)
	}
	// MeanAbs across the layer: (2+1)/4 elements = 0.75.
	if float64(row.MeanAbs) != 0.75 {
		t.Fatalf("layer mean|Δ| = %v, want 0.75", row.MeanAbs)
	}
	// RelL2 = sqrt(4+1)/sqrt(1+4+9+16+25+36 of the two changed base tensors)
	wantRel := math.Sqrt(5) / math.Sqrt(1+4+9+16)
	if math.Abs(float64(row.RelL2)-wantRel) > 1e-12 {
		t.Fatalf("layer relL2 = %v, want %v", row.RelL2, wantRel)
	}
}

func TestGroupDepthOverride(t *testing.T) {
	a := cp(t, f32("model.layers.0.attn.w", 1))
	b := cp(t, f32("model.layers.0.attn.w", 2))
	rep, err := Checkpoints(a, b, Options{GroupDepth: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Layers) != 1 || rep.Layers[0].Layer != "model.layers" {
		t.Fatalf("group depth ignored: %+v", rep.Layers)
	}
	if rep.Tensors[0].Layer != "model.layers" {
		t.Fatalf("tensor layer key must honor depth: %+v", rep.Tensors[0])
	}
}

func TestFilterLimitsComparison(t *testing.T) {
	a := cp(t, f32("model.layers.0.w", 1), f32("model.layers.1.w", 2))
	b := cp(t, f32("model.layers.0.w", 9), f32("model.layers.1.w", 8))
	filter, err := NewFilter([]string{"model.layers.0.*"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := Checkpoints(a, b, Options{Filter: filter})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Summary.Compared != 1 || rep.Summary.Changed != 1 {
		t.Fatalf("filter not applied: %+v", rep.Summary)
	}
	if rep.Summary.BytesA != 4 {
		t.Fatalf("bytes must only count considered tensors: %+v", rep.Summary)
	}
}

func TestReportOrderIsNaturalAndDeterministic(t *testing.T) {
	mkA := func() *checkpoint.Checkpoint {
		return cp(t, f32("model.layers.10.w", 1), f32("model.layers.2.w", 2), f32("model.layers.1.w", 3))
	}
	mkB := func() *checkpoint.Checkpoint {
		return cp(t, f32("model.layers.10.w", 9), f32("model.layers.2.w", 8), f32("model.layers.1.w", 7))
	}
	one, err := Checkpoints(mkA(), mkB(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, td := range one.Tensors {
		names = append(names, td.Name)
	}
	want := []string{"model.layers.1.w", "model.layers.2.w", "model.layers.10.w"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("order = %v, want %v", names, want)
	}
	two, err := Checkpoints(mkA(), mkB(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one.Tensors, two.Tensors) || !reflect.DeepEqual(one.Layers, two.Layers) {
		t.Fatal("same inputs produced different reports")
	}
}

func TestManifestsDiff(t *testing.T) {
	a := cp(t, f32("w.same", 1), f32("w.diff", 2), f32("w.gone", 3))
	b := cp(t, f32("w.same", 1), f32("w.diff", 5), f32("w.new", 4))
	ma, err := manifest.Build(a)
	if err != nil {
		t.Fatal(err)
	}
	mb, err := manifest.Build(b)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := Manifests(ma, mb, "a.json", "b.json", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.HashOnly {
		t.Fatal("manifest diff must be hash-only")
	}
	got := classes(rep)
	want := map[string]Class{"w.diff": ClassChanged, "w.gone": ClassRemoved, "w.new": ClassAdded}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("classes = %v, want %v", got, want)
	}
	if rep.Summary.Identical != 1 {
		t.Fatalf("summary: %+v", rep.Summary)
	}
	for _, td := range rep.Tensors {
		if td.Metrics != nil {
			t.Fatal("hash-only diff must not carry metrics")
		}
	}
}

func TestManifestsDiffDetectsShapeMismatchWithoutData(t *testing.T) {
	// Two tensors with the same bytes but reshaped: a manifest still has
	// the shape, so the mismatch is caught with no weights on disk.
	a := cp(t, f32("w", 1, 2))
	b := cp(t, stwrite.Tensor{Name: "w", DType: "F32", Shape: []int64{1, 2}, Data: stwrite.F32(1, 2)})
	ma, _ := manifest.Build(a)
	mb, _ := manifest.Build(b)
	rep, err := Manifests(ma, mb, "a", "b", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if classes(rep)["w"] != ClassShapeMismatch {
		t.Fatalf("classes = %v", classes(rep))
	}
}

func TestOpaqueDTypeDiffIsChangeDetectedByBytes(t *testing.T) {
	a := cp(t, stwrite.Tensor{Name: "w", DType: "F8_E4M3", Shape: []int64{2}, Data: []byte{1, 2}})
	b := cp(t, stwrite.Tensor{Name: "w", DType: "F8_E4M3", Shape: []int64{2}, Data: []byte{1, 9}})
	rep, err := Checkpoints(a, b, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if classes(rep)["w"] != ClassChanged {
		t.Fatalf("opaque change missed: %v", classes(rep))
	}
	if rep.Tensors[0].Metrics != nil {
		t.Fatal("opaque tensors must not carry numeric metrics")
	}
}
