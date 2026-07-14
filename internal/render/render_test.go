// Tests for rendering: the text report must carry the numbers users
// grep for, the JSON envelope must stay parseable even with NaN stats,
// and the helpers must format edge values sanely.
package render

import (
	"bytes"
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/diff"
	"github.com/JaydenCJ/layerdiff/internal/scan"
	"github.com/JaydenCJ/layerdiff/internal/stwrite"
)

// report builds a real diff report from two small fixtures.
func report(t *testing.T) *diff.Report {
	t.Helper()
	dir := t.TempDir()
	pa := filepath.Join(dir, "a.safetensors")
	pb := filepath.Join(dir, "b.safetensors")
	if err := stwrite.WriteFile(pa, nil, []stwrite.Tensor{
		{Name: "model.layers.0.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 2)},
		{Name: "model.layers.1.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(3, 4)},
		{Name: "gone.w", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(5)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := stwrite.WriteFile(pb, nil, []stwrite.Tensor{
		{Name: "model.layers.0.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 2)},
		{Name: "model.layers.1.w", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(3, 4.5)},
		{Name: "new.w", DType: "F32", Shape: []int64{1}, Data: stwrite.F32(6)},
	}); err != nil {
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
	rep, err := diff.Checkpoints(ca, cb, diff.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

func TestTextReportContents(t *testing.T) {
	var buf bytes.Buffer
	Text(&buf, report(t), 0)
	out := buf.String()
	for _, want := range []string{
		"1 identical", "1 changed", "1 added", "1 removed",
		"changed layers (3 of 4)", "model.layers.1",
		"added tensors (1)", "new.w",
		"removed tensors (1)", "gone.w",
		"verdict: DIFFERENT",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("text report missing %q:\n%s", want, out)
		}
	}

	// The opposite verdict: a report with no differences.
	var same bytes.Buffer
	Text(&same, &diff.Report{PathA: "a", PathB: "b"}, 0)
	if !strings.Contains(same.String(), "verdict: IDENTICAL") {
		t.Fatalf("empty diff must be IDENTICAL:\n%s", same.String())
	}
}

func TestTextTopLimitsChangedRows(t *testing.T) {
	rep := report(t)
	// Duplicate the changed tensor to have 2 rows, then cap at 1.
	rep.Tensors = append(rep.Tensors, rep.Tensors...)
	var buf bytes.Buffer
	Text(&buf, rep, 1)
	if !strings.Contains(buf.String(), "(1 of 2 shown") {
		t.Fatalf("--top cap not reflected:\n%s", buf.String())
	}
}

func TestJSONEnvelopeAndRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := JSON(&buf, report(t)); err != nil {
		t.Fatalf("JSON: %v", err)
	}
	var doc struct {
		Tool          string `json:"tool"`
		Version       string `json:"version"`
		SchemaVersion int    `json:"schema_version"`
		Summary       struct {
			Changed int `json:"changed"`
		} `json:"summary"`
		Tensors []map[string]any `json:"tensors"`
	}
	if err := json.Unmarshal(buf.Bytes(), &doc); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if doc.Tool != "layerdiff" || doc.SchemaVersion != 1 || doc.Version == "" {
		t.Fatalf("envelope wrong: %+v", doc)
	}
	if doc.Summary.Changed != 1 || len(doc.Tensors) != 3 {
		t.Fatalf("payload wrong: %+v", doc)
	}
}

func TestJSONSurvivesNonFiniteValues(t *testing.T) {
	rep := report(t)
	rep.Layers[0].RelL2 = scan.F64(math.NaN())
	rep.Layers[0].MaxAbs = scan.F64(math.Inf(1))
	var buf bytes.Buffer
	if err := JSON(&buf, rep); err != nil {
		t.Fatalf("NaN/Inf must not break JSON rendering: %v", err)
	}
	if !strings.Contains(buf.String(), `"NaN"`) || !strings.Contains(buf.String(), `"Infinity"`) {
		t.Fatalf("non-finite encoding missing:\n%s", buf.String())
	}
}

func TestMarkdownReport(t *testing.T) {
	var buf bytes.Buffer
	Markdown(&buf, report(t), 0)
	out := buf.String()
	for _, want := range []string{
		"## layerdiff:", "| compared |", "### Changed layers",
		"### Differing tensors", "`model.layers.1.w`", "**Verdict: DIFFERENT**",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("markdown missing %q:\n%s", want, out)
		}
	}
}

func TestListingTextAndJSON(t *testing.T) {
	l := &Listing{
		Path: "m.safetensors", Kind: "safetensors", Shards: []string{"m.safetensors"},
		Metadata:   map[string]string{"format": "pt"},
		TotalBytes: 8,
		Tensors: []ListEntry{{
			Name: "w", DType: "F32", Shape: []int64{2}, Elems: 2, Bytes: 8,
			SHA256: strings.Repeat("ab", 32),
		}},
	}
	var text bytes.Buffer
	ListingText(&text, l, true, false)
	out := text.String()
	// Counts of one read singular: "1 tensor", "1 shard".
	if !strings.Contains(out, "1 tensor ·") || !strings.Contains(out, "1 shard\n") ||
		!strings.Contains(out, "format=pt") || !strings.Contains(out, "abababababab") {
		t.Fatalf("listing text wrong:\n%s", out)
	}
	var js bytes.Buffer
	if err := ListingJSON(&js, l); err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Tool    string `json:"tool"`
		Tensors []struct {
			Name string `json:"name"`
		} `json:"tensors"`
	}
	if err := json.Unmarshal(js.Bytes(), &doc); err != nil {
		t.Fatalf("listing JSON invalid: %v", err)
	}
	if doc.Tool != "layerdiff" || len(doc.Tensors) != 1 || doc.Tensors[0].Name != "w" {
		t.Fatalf("listing JSON wrong: %+v", doc)
	}
}

func TestFormattingHelpers(t *testing.T) {
	cases := map[int64]string{
		0:              "0 B",
		512:            "512 B",
		2048:           "2.0 KiB",
		5 << 20:        "5.0 MiB",
		3 << 30:        "3.0 GiB",
		1536 * 1 << 30: "1.5 TiB",
	}
	for in, want := range cases {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
	if got := sci(scan.F64(0.0123)); got != "1.23e-02" {
		t.Errorf("sci = %q", got)
	}
	if got := sci(scan.F64(math.NaN())); got != "-" {
		t.Errorf("sci NaN = %q", got)
	}
	if got := sci(scan.F64(math.Inf(1))); got != "inf" {
		t.Errorf("sci Inf = %q", got)
	}
	if got := sci(scan.F64(0)); got != "0" {
		t.Errorf("sci 0 = %q", got)
	}
	if got := shapeString([]int64{4096, 128}); got != "[4096,128]" {
		t.Errorf("shapeString = %q", got)
	}
	if got := shapeString(nil); got != "[]" {
		t.Errorf("scalar shapeString = %q", got)
	}
}
