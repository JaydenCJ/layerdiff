// Shared formatting helpers: aligned tables, human-readable byte sizes,
// compact scientific notation, and shape strings.
package render

import (
	"fmt"
	"io"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/JaydenCJ/layerdiff/internal/scan"
)

// table accumulates rows and prints them with per-column alignment.
type table struct {
	rows  [][]string
	right []bool // per-column right alignment
}

func (t *table) add(cells ...string) {
	t.rows = append(t.rows, cells)
}

func (t *table) write(w io.Writer, indent string) {
	widths := make([]int, 0)
	for _, row := range t.rows {
		for i, c := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			if n := utf8.RuneCountInString(c); n > widths[i] {
				widths[i] = n
			}
		}
	}
	for _, row := range t.rows {
		var b strings.Builder
		b.WriteString(indent)
		for i, c := range row {
			if i > 0 {
				b.WriteString("  ")
			}
			pad := widths[i] - utf8.RuneCountInString(c)
			last := i == len(row)-1
			if i < len(t.right) && t.right[i] {
				b.WriteString(strings.Repeat(" ", pad))
				b.WriteString(c)
			} else {
				b.WriteString(c)
				if !last {
					b.WriteString(strings.Repeat(" ", pad))
				}
			}
		}
		fmt.Fprintln(w, strings.TrimRight(b.String(), " "))
	}
}

// humanBytes renders a byte count in binary units.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// sci renders a magnitude in compact scientific/short notation; NaN
// (used for "not applicable") renders as a dash.
func sci(v scan.F64) string {
	f := float64(v)
	switch {
	case math.IsNaN(f):
		return "-"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case f == 0:
		return "0"
	}
	return fmt.Sprintf("%.2e", f)
}

// fixed renders a value with limited precision for stats columns.
func fixed(v scan.F64) string {
	f := float64(v)
	switch {
	case math.IsNaN(f):
		return "-"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	return fmt.Sprintf("%.4g", f)
}

// shapeString renders a shape as [d0,d1,…]; a scalar is [].
func shapeString(shape []int64) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// plural returns "" for a count of one and "s" otherwise, for simple
// count phrases like "2 shards".
func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// short truncates a hex digest for terminal display.
func short(hash string) string {
	if len(hash) > 12 {
		return hash[:12]
	}
	return hash
}
