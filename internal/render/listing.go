// Listing rendering for `layerdiff ls`: the inventory of one checkpoint,
// optionally with per-tensor hashes and streamed statistics.
package render

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/JaydenCJ/layerdiff/internal/scan"
	"github.com/JaydenCJ/layerdiff/internal/version"
)

// Listing is the model behind `layerdiff ls`.
type Listing struct {
	Path       string            `json:"path"`
	Kind       string            `json:"kind"`
	Shards     []string          `json:"shards"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	TotalBytes int64             `json:"total_bytes"`
	Tensors    []ListEntry       `json:"tensors"`
}

// ListEntry is one tensor row of a listing.
type ListEntry struct {
	Name   string      `json:"name"`
	DType  string      `json:"dtype"`
	Shape  []int64     `json:"shape"`
	Elems  int64       `json:"elems"`
	Bytes  int64       `json:"bytes"`
	SHA256 string      `json:"sha256,omitempty"`
	Stats  *scan.Stats `json:"stats,omitempty"`
}

// ListingText writes the terminal view.
func ListingText(w io.Writer, l *Listing, withHash, withStats bool) {
	fmt.Fprintf(w, "%s — %s checkpoint, %d shard%s\n", l.Path, l.Kind, len(l.Shards), plural(len(l.Shards)))
	if len(l.Metadata) > 0 {
		keys := make([]string, 0, len(l.Metadata))
		for k := range l.Metadata {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Fprint(w, "metadata:")
		for _, k := range keys {
			fmt.Fprintf(w, " %s=%s", k, l.Metadata[k])
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "%d tensor%s · %s tensor data\n\n", len(l.Tensors), plural(len(l.Tensors)), humanBytes(l.TotalBytes))

	t := &table{right: []bool{false, false, false, true, true}}
	head := []string{"tensor", "dtype", "shape", "elems", "bytes"}
	if withHash {
		head = append(head, "sha256")
		t.right = append(t.right, false)
	}
	if withStats {
		head = append(head, "min", "max", "mean", "rms")
		t.right = append(t.right, true, true, true, true)
	}
	t.add(head...)
	for _, e := range l.Tensors {
		row := []string{e.Name, e.DType, shapeString(e.Shape),
			fmt.Sprintf("%d", e.Elems), fmt.Sprintf("%d", e.Bytes)}
		if withHash {
			row = append(row, short(e.SHA256))
		}
		if withStats {
			if e.Stats != nil {
				row = append(row, fixed(e.Stats.Min), fixed(e.Stats.Max), fixed(e.Stats.Mean), fixed(e.Stats.RMS))
			} else {
				row = append(row, "-", "-", "-", "-")
			}
		}
		t.add(row...)
	}
	t.write(w, "  ")
}

// ListingJSON writes the machine view, wrapped in the same envelope as
// diff reports.
func ListingJSON(w io.Writer, l *Listing) error {
	out, err := json.MarshalIndent(struct {
		Tool          string `json:"tool"`
		Version       string `json:"version"`
		SchemaVersion int    `json:"schema_version"`
		*Listing
	}{"layerdiff", version.Version, SchemaVersion, l}, "", "  ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(w, string(out))
	return err
}
