// Terminal rendering of a diff report: summary counts, the changed-layer
// rollup, and per-class tensor tables, all deterministically ordered.
package render

import (
	"fmt"
	"io"
	"sort"

	"github.com/JaydenCJ/layerdiff/internal/diff"
	"github.com/JaydenCJ/layerdiff/internal/layer"
)

// Text writes the human-readable report. top limits the changed-tensor
// table (0 = unlimited); JSON output is never truncated.
func Text(w io.Writer, r *diff.Report, top int) {
	fmt.Fprintf(w, "layerdiff — %s → %s\n", r.PathA, r.PathB)
	if r.HashOnly {
		fmt.Fprintln(w, "mode: hash-only (manifest identity, no elementwise metrics)")
	}
	if r.Atol != 0 || r.Rtol != 0 {
		fmt.Fprintf(w, "tolerance: atol=%g rtol=%g\n", r.Atol, r.Rtol)
	}
	s := r.Summary
	fmt.Fprintf(w, "tensors: %d compared · %d identical · %d equivalent · %d changed · %d added · %d removed · %d mismatched\n",
		s.Compared, s.Identical, s.Equivalent, s.Changed, s.Added, s.Removed, s.Mismatched)
	if r.HashOnly {
		fmt.Fprintf(w, "data: %s vs %s of tensor data\n", humanBytes(s.BytesA), humanBytes(s.BytesB))
	} else {
		fmt.Fprintf(w, "data: %s vs %s, streamed in constant memory\n", humanBytes(s.BytesA), humanBytes(s.BytesB))
	}

	if len(r.Layers) > 0 {
		fmt.Fprintf(w, "\nchanged layers (%d of %d)\n", s.LayersChanged, s.Layers)
		t := &table{right: []bool{false, true, true, true, true, true, true, true}}
		t.add("layer", "tensors", "changed", "added", "removed", "max|Δ|", "mean|Δ|", "rel L2")
		for _, row := range r.Layers {
			t.add(row.Layer,
				fmt.Sprintf("%d", row.Tensors),
				fmt.Sprintf("%d", row.Changed+row.Mismatched),
				fmt.Sprintf("%d", row.Added),
				fmt.Sprintf("%d", row.Removed),
				sci(row.MaxAbs), sci(row.MeanAbs), sci(row.RelL2))
		}
		t.write(w, "  ")
	}

	writeChanged(w, r, top)
	writeClassTable(w, r, diff.ClassEquivalent, "equivalent tensors (within tolerance)")
	writeMismatched(w, r)
	writeOneSided(w, r, diff.ClassAdded, "added tensors")
	writeOneSided(w, r, diff.ClassRemoved, "removed tensors")

	fmt.Fprintf(w, "\nverdict: %s\n", verdict(r))
}

func verdict(r *diff.Report) string {
	switch {
	case r.Different():
		return "DIFFERENT"
	case r.Summary.Equivalent > 0:
		return "EQUIVALENT (differences within tolerance)"
	default:
		return "IDENTICAL"
	}
}

// changedSorted returns changed tensors ordered by descending max|Δ|,
// with natural name order breaking ties (and standing in entirely for
// hash-only reports, which have no magnitudes).
func changedSorted(r *diff.Report) []diff.TensorDiff {
	var rows []diff.TensorDiff
	for _, td := range r.Tensors {
		if td.Class == diff.ClassChanged {
			rows = append(rows, td)
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		mi, mj := rows[i].Metrics, rows[j].Metrics
		if mi != nil && mj != nil && mi.MaxAbs != mj.MaxAbs {
			return mi.MaxAbs > mj.MaxAbs
		}
		return layer.NaturalLess(rows[i].Name, rows[j].Name)
	})
	return rows
}

func writeChanged(w io.Writer, r *diff.Report, top int) {
	rows := changedSorted(r)
	if len(rows) == 0 {
		return
	}
	shown := len(rows)
	if top > 0 && top < shown {
		shown = top
	}
	if r.HashOnly {
		fmt.Fprintf(w, "\nchanged tensors (%d of %d shown)\n", shown, len(rows))
		t := &table{right: []bool{false, false, false, true, false, false}}
		t.add("tensor", "dtype", "shape", "bytes", "sha256 A", "sha256 B")
		for _, td := range rows[:shown] {
			t.add(td.Name, td.DTypeA, shapeString(td.ShapeA), humanBytes(td.BytesA),
				short(td.SHA256A), short(td.SHA256B))
		}
		t.write(w, "  ")
		return
	}
	fmt.Fprintf(w, "\nchanged tensors (%d of %d shown, by max|Δ|)\n", shown, len(rows))
	t := &table{right: []bool{false, false, false, true, true, true, true}}
	t.add("tensor", "dtype", "shape", "max|Δ|", "mean|Δ|", "changed elems", "cosine")
	for _, td := range rows[:shown] {
		maxAbs, meanAbs, changed, cos := "-", "-", "-", "-"
		if m := td.Metrics; m != nil {
			maxAbs, meanAbs = sci(m.MaxAbs), sci(m.MeanAbs)
			changed = fmt.Sprintf("%d/%d", m.Changed, m.Elems)
			cos = fixed(m.Cosine)
		}
		t.add(td.Name, td.DTypeA, shapeString(td.ShapeA), maxAbs, meanAbs, changed, cos)
	}
	t.write(w, "  ")
}

func writeClassTable(w io.Writer, r *diff.Report, class diff.Class, title string) {
	var rows []diff.TensorDiff
	for _, td := range r.Tensors {
		if td.Class == class {
			rows = append(rows, td)
		}
	}
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%s (%d)\n", title, len(rows))
	t := &table{right: []bool{false, false, false, true}}
	t.add("tensor", "dtype", "shape", "max|Δ|")
	for _, td := range rows {
		maxAbs := "-"
		if td.Metrics != nil {
			maxAbs = sci(td.Metrics.MaxAbs)
		}
		t.add(td.Name, td.DTypeA, shapeString(td.ShapeA), maxAbs)
	}
	t.write(w, "  ")
}

func writeMismatched(w io.Writer, r *diff.Report) {
	var rows []diff.TensorDiff
	for _, td := range r.Tensors {
		if td.Class == diff.ClassDTypeMismatch || td.Class == diff.ClassShapeMismatch {
			rows = append(rows, td)
		}
	}
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\nmismatched tensors (%d)\n", len(rows))
	t := &table{}
	t.add("tensor", "kind", "A", "B")
	for _, td := range rows {
		kind := "shape"
		if td.Class == diff.ClassDTypeMismatch {
			kind = "dtype"
		}
		t.add(td.Name, kind,
			td.DTypeA+" "+shapeString(td.ShapeA),
			td.DTypeB+" "+shapeString(td.ShapeB))
	}
	t.write(w, "  ")
}

func writeOneSided(w io.Writer, r *diff.Report, class diff.Class, title string) {
	var rows []diff.TensorDiff
	for _, td := range r.Tensors {
		if td.Class == class {
			rows = append(rows, td)
		}
	}
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%s (%d)\n", title, len(rows))
	t := &table{right: []bool{false, false, false, true}}
	t.add("tensor", "dtype", "shape", "bytes")
	for _, td := range rows {
		dt, shape, bytes := td.DTypeA, td.ShapeA, td.BytesA
		if class == diff.ClassAdded {
			dt, shape, bytes = td.DTypeB, td.ShapeB, td.BytesB
		}
		t.add(td.Name, dt, shapeString(shape), humanBytes(bytes))
	}
	t.write(w, "  ")
}
