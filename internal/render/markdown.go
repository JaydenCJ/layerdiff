// Markdown rendering: paste-ready tables for PRs, model cards, and merge
// notes.
package render

import (
	"fmt"
	"io"
	"strings"

	"github.com/JaydenCJ/layerdiff/internal/diff"
)

// Markdown writes the report as GitHub-flavored Markdown. top limits the
// differing-tensor table (0 = unlimited).
func Markdown(w io.Writer, r *diff.Report, top int) {
	fmt.Fprintf(w, "## layerdiff: `%s` → `%s`\n\n", r.PathA, r.PathB)
	if r.HashOnly {
		fmt.Fprintf(w, "Hash-only comparison (manifest identity, no elementwise metrics).\n\n")
	}
	if r.Atol != 0 || r.Rtol != 0 {
		fmt.Fprintf(w, "Tolerance: `atol=%g rtol=%g`.\n\n", r.Atol, r.Rtol)
	}

	s := r.Summary
	fmt.Fprintln(w, "| compared | identical | equivalent | changed | added | removed | mismatched |")
	fmt.Fprintln(w, "|---:|---:|---:|---:|---:|---:|---:|")
	fmt.Fprintf(w, "| %d | %d | %d | %d | %d | %d | %d |\n\n",
		s.Compared, s.Identical, s.Equivalent, s.Changed, s.Added, s.Removed, s.Mismatched)

	if len(r.Layers) > 0 {
		fmt.Fprintf(w, "### Changed layers (%d of %d)\n\n", s.LayersChanged, s.Layers)
		fmt.Fprintln(w, "| layer | tensors | changed | added | removed | max\\|Δ\\| | mean\\|Δ\\| | rel L2 |")
		fmt.Fprintln(w, "|---|---:|---:|---:|---:|---:|---:|---:|")
		for _, row := range r.Layers {
			fmt.Fprintf(w, "| `%s` | %d | %d | %d | %d | %s | %s | %s |\n",
				row.Layer, row.Tensors, row.Changed+row.Mismatched, row.Added, row.Removed,
				sci(row.MaxAbs), sci(row.MeanAbs), sci(row.RelL2))
		}
		fmt.Fprintln(w)
	}

	if len(r.Tensors) > 0 {
		rows := append(changedSorted(r), otherDiffs(r)...)
		shown := len(rows)
		if top > 0 && top < shown {
			shown = top
		}
		fmt.Fprintf(w, "### Differing tensors (%d of %d shown)\n\n", shown, len(rows))
		fmt.Fprintln(w, "| tensor | class | dtype | shape | max\\|Δ\\| | changed elems |")
		fmt.Fprintln(w, "|---|---|---|---|---:|---:|")
		for _, td := range rows[:shown] {
			dt, shape := td.DTypeA, td.ShapeA
			if td.Class == diff.ClassAdded {
				dt, shape = td.DTypeB, td.ShapeB
			}
			maxAbs, changed := "—", "—"
			if m := td.Metrics; m != nil {
				maxAbs = sci(m.MaxAbs)
				changed = fmt.Sprintf("%d/%d", m.Changed, m.Elems)
			}
			fmt.Fprintf(w, "| `%s` | %s | %s | `%s` | %s | %s |\n",
				td.Name, td.Class, dt, strings.ReplaceAll(shapeString(shape), "|", "\\|"), maxAbs, changed)
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintf(w, "**Verdict: %s**\n", verdict(r))
}

// otherDiffs returns the non-changed, non-identical rows (equivalent,
// added, removed, mismatched) in report order.
func otherDiffs(r *diff.Report) []diff.TensorDiff {
	var rows []diff.TensorDiff
	for _, td := range r.Tensors {
		if td.Class != diff.ClassChanged {
			rows = append(rows, td)
		}
	}
	return rows
}
