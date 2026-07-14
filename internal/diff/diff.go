// Package diff orchestrates the tensor-by-tensor comparison of two
// checkpoints (or hash manifests) into a Report: a summary, a per-layer
// rollup of where change concentrated, and one entry per differing
// tensor. All ordering is deterministic (natural name order), so
// identical inputs always produce byte-identical reports.
package diff

import (
	"math"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/layer"
	"github.com/JaydenCJ/layerdiff/internal/manifest"
	"github.com/JaydenCJ/layerdiff/internal/scan"
)

// Class labels how one tensor compares across the two sides.
type Class string

const (
	// ClassIdentical: byte-for-byte equal (same hash).
	ClassIdentical Class = "identical"
	// ClassEquivalent: bytes differ but no element exceeds the tolerance
	// (or the values are numerically equal, e.g. -0.0 vs 0.0).
	ClassEquivalent Class = "equivalent"
	// ClassChanged: at least one element differs beyond the tolerance.
	ClassChanged Class = "changed"
	// ClassAdded / ClassRemoved: present on only one side.
	ClassAdded   Class = "added"
	ClassRemoved Class = "removed"
	// Mismatches: present on both sides but not elementwise comparable.
	ClassDTypeMismatch Class = "dtype-mismatch"
	ClassShapeMismatch Class = "shape-mismatch"
)

// Options tune a comparison.
type Options struct {
	Tolerance scan.Tolerance
	// GroupDepth overrides the automatic layer-key heuristic; see
	// layer.Key.
	GroupDepth int
	// Filter restricts which tensor names are compared; nil means all.
	Filter *Filter
}

// TensorDiff is one differing tensor. Identical tensors are counted in
// the summary but not listed.
type TensorDiff struct {
	Name    string            `json:"name"`
	Layer   string            `json:"layer"`
	Class   Class             `json:"class"`
	DTypeA  string            `json:"dtype_a,omitempty"`
	DTypeB  string            `json:"dtype_b,omitempty"`
	ShapeA  []int64           `json:"shape_a,omitempty"`
	ShapeB  []int64           `json:"shape_b,omitempty"`
	BytesA  int64             `json:"bytes_a,omitempty"`
	BytesB  int64             `json:"bytes_b,omitempty"`
	SHA256A string            `json:"sha256_a,omitempty"`
	SHA256B string            `json:"sha256_b,omitempty"`
	StatsA  *scan.Stats       `json:"stats_a,omitempty"`
	StatsB  *scan.Stats       `json:"stats_b,omitempty"`
	Metrics *scan.DiffMetrics `json:"metrics,omitempty"`
}

// LayerRow aggregates one layer group. Magnitude columns are NaN when no
// numeric metrics exist for the layer (hash-only diffs, opaque dtypes).
type LayerRow struct {
	Layer      string   `json:"layer"`
	Tensors    int      `json:"tensors"`
	Identical  int      `json:"identical"`
	Equivalent int      `json:"equivalent"`
	Changed    int      `json:"changed"`
	Added      int      `json:"added"`
	Removed    int      `json:"removed"`
	Mismatched int      `json:"mismatched"`
	MaxAbs     scan.F64 `json:"max_abs"`
	MeanAbs    scan.F64 `json:"mean_abs"`
	RelL2      scan.F64 `json:"rel_l2"`
}

// Summary counts the whole comparison.
type Summary struct {
	TensorsA      int   `json:"tensors_a"`
	TensorsB      int   `json:"tensors_b"`
	Compared      int   `json:"compared"`
	Identical     int   `json:"identical"`
	Equivalent    int   `json:"equivalent"`
	Changed       int   `json:"changed"`
	Added         int   `json:"added"`
	Removed       int   `json:"removed"`
	Mismatched    int   `json:"mismatched"`
	Layers        int   `json:"layers"`
	LayersChanged int   `json:"layers_changed"`
	BytesA        int64 `json:"bytes_a"`
	BytesB        int64 `json:"bytes_b"`
}

// Report is the complete outcome of a comparison.
type Report struct {
	PathA    string       `json:"path_a"`
	PathB    string       `json:"path_b"`
	KindA    string       `json:"kind_a"`
	KindB    string       `json:"kind_b"`
	HashOnly bool         `json:"hash_only"`
	Atol     float64      `json:"atol"`
	Rtol     float64      `json:"rtol"`
	Summary  Summary      `json:"summary"`
	Layers   []LayerRow   `json:"layers"`
	Tensors  []TensorDiff `json:"tensors"`
}

// Different reports whether the two sides differ beyond tolerance —
// this drives the process exit code.
func (r *Report) Different() bool {
	s := r.Summary
	return s.Changed+s.Added+s.Removed+s.Mismatched > 0
}

// Checkpoints compares two resolved checkpoints, streaming every
// considered tensor exactly once per side.
func Checkpoints(a, b *checkpoint.Checkpoint, opts Options) (*Report, error) {
	rep := &Report{
		PathA: a.Path, PathB: b.Path,
		KindA: a.Kind, KindB: b.Kind,
		Atol: opts.Tolerance.Atol, Rtol: opts.Tolerance.Rtol,
	}
	agg := newAggregator(rep)

	for _, name := range unionNames(a.Names(), b.Names(), opts.Filter) {
		ta, inA := a.Tensor(name)
		tb, inB := b.Tensor(name)
		td := TensorDiff{Name: name, Layer: layer.Key(name, opts.GroupDepth)}

		switch {
		case inA && !inB:
			td.Class = ClassRemoved
			if err := fillSide(&td, ta, nil); err != nil {
				return nil, err
			}
		case !inA && inB:
			td.Class = ClassAdded
			if err := fillSide(&td, nil, tb); err != nil {
				return nil, err
			}
		case ta.DType != tb.DType:
			td.Class = ClassDTypeMismatch
			if err := fillSide(&td, ta, tb); err != nil {
				return nil, err
			}
		case !shapeEqual(ta.Shape, tb.Shape):
			td.Class = ClassShapeMismatch
			if err := fillSide(&td, ta, tb); err != nil {
				return nil, err
			}
		default:
			pr, err := scan.Pair(ta, tb, opts.Tolerance)
			if err != nil {
				return nil, err
			}
			td.DTypeA, td.DTypeB = ta.DType, tb.DType
			td.ShapeA, td.ShapeB = ta.Shape, tb.Shape
			td.BytesA, td.BytesB = ta.Bytes, tb.Bytes
			td.SHA256A, td.SHA256B = pr.A.SHA256, pr.B.SHA256
			td.StatsA, td.StatsB = pr.A.Stats, pr.B.Stats
			td.Metrics = pr.Metrics
			switch {
			case pr.BitwiseEqual:
				td.Class = ClassIdentical
			case pr.Metrics != nil && pr.Metrics.Changed == 0:
				td.Class = ClassEquivalent
			default:
				td.Class = ClassChanged
			}
		}
		agg.add(td)
	}

	agg.finish()
	return rep, nil
}

// Manifests compares two hash manifests: identity only, no elementwise
// metrics. pathA/pathB label the report with what the user passed.
func Manifests(a, b *manifest.Manifest, pathA, pathB string, opts Options) (*Report, error) {
	rep := &Report{
		PathA: pathA, PathB: pathB,
		KindA: "manifest", KindB: "manifest",
		HashOnly: true,
		Atol:     opts.Tolerance.Atol, Rtol: opts.Tolerance.Rtol,
	}
	agg := newAggregator(rep)

	namesA := make([]string, 0, len(a.Tensors))
	for n := range a.Tensors {
		namesA = append(namesA, n)
	}
	namesB := make([]string, 0, len(b.Tensors))
	for n := range b.Tensors {
		namesB = append(namesB, n)
	}

	for _, name := range unionNames(namesA, namesB, opts.Filter) {
		ea, inA := a.Tensors[name]
		eb, inB := b.Tensors[name]
		td := TensorDiff{Name: name, Layer: layer.Key(name, opts.GroupDepth)}
		switch {
		case inA && !inB:
			td.Class = ClassRemoved
			td.DTypeA, td.ShapeA, td.BytesA, td.SHA256A = ea.DType, ea.Shape, ea.Bytes, ea.SHA256
		case !inA && inB:
			td.Class = ClassAdded
			td.DTypeB, td.ShapeB, td.BytesB, td.SHA256B = eb.DType, eb.Shape, eb.Bytes, eb.SHA256
		default:
			td.DTypeA, td.ShapeA, td.BytesA, td.SHA256A = ea.DType, ea.Shape, ea.Bytes, ea.SHA256
			td.DTypeB, td.ShapeB, td.BytesB, td.SHA256B = eb.DType, eb.Shape, eb.Bytes, eb.SHA256
			switch {
			case ea.DType != eb.DType:
				td.Class = ClassDTypeMismatch
			case !shapeEqual(ea.Shape, eb.Shape):
				td.Class = ClassShapeMismatch
			case ea.SHA256 == eb.SHA256:
				td.Class = ClassIdentical
			default:
				td.Class = ClassChanged
			}
		}
		agg.add(td)
	}

	agg.finish()
	return rep, nil
}

// fillSide scans whichever sides are present individually (used for
// added/removed tensors and dtype/shape mismatches, where lockstep
// comparison is impossible).
func fillSide(td *TensorDiff, ta, tb *checkpoint.Tensor) error {
	if ta != nil {
		res, err := scan.Tensor(ta)
		if err != nil {
			return err
		}
		td.DTypeA, td.ShapeA, td.BytesA = ta.DType, ta.Shape, ta.Bytes
		td.SHA256A, td.StatsA = res.SHA256, res.Stats
	}
	if tb != nil {
		res, err := scan.Tensor(tb)
		if err != nil {
			return err
		}
		td.DTypeB, td.ShapeB, td.BytesB = tb.DType, tb.Shape, tb.Bytes
		td.SHA256B, td.StatsB = res.SHA256, res.Stats
	}
	return nil
}

func shapeEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// unionNames merges, filters, and naturally sorts the two name sets.
func unionNames(a, b []string, f *Filter) []string {
	seen := make(map[string]bool, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, list := range [][]string{a, b} {
		for _, n := range list {
			if !seen[n] && f.Match(n) {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	layer.SortNames(out)
	return out
}

// aggregator folds TensorDiffs into the report's summary and layer rows.
type aggregator struct {
	rep    *Report
	layers map[string]*layerAcc
	keys   []string
}

type layerAcc struct {
	row        LayerRow
	maxAbs     float64
	sumAbs     float64
	numeric    int64
	sumSqDiff  float64
	sumSqBase  float64
	hasMetrics bool
}

func newAggregator(rep *Report) *aggregator {
	return &aggregator{rep: rep, layers: make(map[string]*layerAcc)}
}

func (g *aggregator) add(td TensorDiff) {
	s := &g.rep.Summary
	s.Compared++
	if td.Class != ClassAdded {
		s.TensorsA++
		s.BytesA += td.BytesA
	}
	if td.Class != ClassRemoved {
		s.TensorsB++
		s.BytesB += td.BytesB
	}

	acc, ok := g.layers[td.Layer]
	if !ok {
		acc = &layerAcc{row: LayerRow{Layer: td.Layer}}
		g.layers[td.Layer] = acc
		g.keys = append(g.keys, td.Layer)
	}
	acc.row.Tensors++

	switch td.Class {
	case ClassIdentical:
		s.Identical++
		acc.row.Identical++
	case ClassEquivalent:
		s.Equivalent++
		acc.row.Equivalent++
	case ClassChanged:
		s.Changed++
		acc.row.Changed++
	case ClassAdded:
		s.Added++
		acc.row.Added++
	case ClassRemoved:
		s.Removed++
		acc.row.Removed++
	case ClassDTypeMismatch, ClassShapeMismatch:
		s.Mismatched++
		acc.row.Mismatched++
	}

	if m := td.Metrics; m != nil && (td.Class == ClassChanged || td.Class == ClassEquivalent) {
		acc.hasMetrics = true
		if v := float64(m.MaxAbs); v > acc.maxAbs {
			acc.maxAbs = v
		}
		acc.sumAbs += float64(m.L1)
		acc.numeric += m.NumericPairs
		acc.sumSqDiff += float64(m.L2) * float64(m.L2)
		if td.StatsA != nil {
			acc.sumSqBase += float64(td.StatsA.L2) * float64(td.StatsA.L2)
		}
	}

	if td.Class != ClassIdentical {
		g.rep.Tensors = append(g.rep.Tensors, td)
	}
}

// finish materializes layer rows (only layers that differ are listed)
// and the layer counts.
func (g *aggregator) finish() {
	layer.SortNames(g.keys)
	g.rep.Summary.Layers = len(g.keys)
	if g.rep.Tensors == nil {
		g.rep.Tensors = []TensorDiff{}
	}
	g.rep.Layers = []LayerRow{}
	for _, key := range g.keys {
		acc := g.layers[key]
		r := acc.row
		if r.Changed+r.Added+r.Removed+r.Mismatched == 0 {
			continue
		}
		g.rep.Summary.LayersChanged++
		nan := scan.F64(math.NaN())
		r.MaxAbs, r.MeanAbs, r.RelL2 = nan, nan, nan
		if acc.hasMetrics {
			r.MaxAbs = scan.F64(acc.maxAbs)
			if acc.numeric > 0 {
				r.MeanAbs = scan.F64(acc.sumAbs / float64(acc.numeric))
			}
			switch {
			case acc.sumSqBase > 0:
				r.RelL2 = scan.F64(math.Sqrt(acc.sumSqDiff) / math.Sqrt(acc.sumSqBase))
			case acc.sumSqDiff == 0:
				r.RelL2 = 0
			default:
				r.RelL2 = scan.F64(math.Inf(1))
			}
		}
		g.rep.Layers = append(g.rep.Layers, r)
	}
}
