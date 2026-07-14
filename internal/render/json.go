// JSON rendering: a stable envelope (tool, version, schema_version 1)
// around the full report. Nothing is truncated here — --top only limits
// the terminal tables.
package render

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/JaydenCJ/layerdiff/internal/diff"
	"github.com/JaydenCJ/layerdiff/internal/version"
)

// SchemaVersion identifies the JSON report layout; it only moves on
// breaking changes.
const SchemaVersion = 1

type envelope struct {
	Tool          string `json:"tool"`
	Version       string `json:"version"`
	SchemaVersion int    `json:"schema_version"`
	*diff.Report
}

// JSON writes the machine-readable report.
func JSON(w io.Writer, r *diff.Report) error {
	out, err := json.MarshalIndent(envelope{
		Tool:          "layerdiff",
		Version:       version.Version,
		SchemaVersion: SchemaVersion,
		Report:        r,
	}, "", "  ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(w, string(out))
	return err
}
