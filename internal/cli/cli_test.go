// In-process CLI integration tests: Run() is exercised exactly as main()
// would, against real checkpoint files fabricated in temp dirs, and the
// documented exit-code contract is pinned down.
package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JaydenCJ/layerdiff/internal/stwrite"
	"github.com/JaydenCJ/layerdiff/internal/version"
)

// run invokes the CLI in-process.
func run(args ...string) (code int, stdout, stderr string) {
	var out, errBuf bytes.Buffer
	code = Run(args, &out, &errBuf)
	return code, out.String(), errBuf.String()
}

// fixturePair writes a base and a tuned checkpoint and returns their paths.
func fixturePair(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	base := filepath.Join(dir, "base.safetensors")
	tuned := filepath.Join(dir, "tuned.safetensors")
	if err := stwrite.WriteFile(base, nil, []stwrite.Tensor{
		{Name: "model.layers.0.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(1, 2, 3, 4)},
		{Name: "model.layers.1.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(5, 6, 7, 8)},
		{Name: "model.norm.weight", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 1)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := stwrite.WriteFile(tuned, nil, []stwrite.Tensor{
		{Name: "model.layers.0.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(1, 2, 3, 4)},
		{Name: "model.layers.1.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(5, 6.5, 7, 8)},
		{Name: "model.norm.weight", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 1)},
	}); err != nil {
		t.Fatal(err)
	}
	return base, tuned
}

func TestVersionAndHelp(t *testing.T) {
	for _, arg := range []string{"version", "--version", "-v"} {
		code, out, _ := run(arg)
		if code != ExitSame || out != "layerdiff "+version.Version+"\n" {
			t.Errorf("%s: code=%d out=%q", arg, code, out)
		}
	}
	code, out, _ := run("help")
	if code != ExitSame {
		t.Fatalf("help exit = %d", code)
	}
	if !strings.Contains(out, "Exit codes") || !strings.Contains(out, "layerdiff diff") {
		t.Fatalf("help output incomplete:\n%s", out)
	}
	// --help on a subcommand is a request for help, not a usage error:
	// it must print the usage to stdout and exit 0.
	for _, sub := range []string{"diff", "ls", "hash"} {
		code, out, stderr := run(sub, "--help")
		if code != ExitSame || !strings.Contains(out, "Exit codes") || stderr != "" {
			t.Errorf("%s --help: code=%d stderr=%q", sub, code, stderr)
		}
	}
}

func TestNoArgsAndUnknownCommandAreUsageErrors(t *testing.T) {
	if code, _, _ := run(); code != ExitUsage {
		t.Fatalf("no args: exit = %d", code)
	}
	code, _, stderr := run("frobnicate")
	if code != ExitUsage || !strings.Contains(stderr, "unknown command") {
		t.Fatalf("unknown command: code=%d stderr=%q", code, stderr)
	}
}

func TestDiffIdenticalExitsZero(t *testing.T) {
	base, _ := fixturePair(t)
	code, out, _ := run("diff", base, base)
	if code != ExitSame {
		t.Fatalf("identical diff exit = %d", code)
	}
	if !strings.Contains(out, "verdict: IDENTICAL") {
		t.Fatalf("missing verdict:\n%s", out)
	}
}

func TestDiffChangedExitsOneWithReport(t *testing.T) {
	base, tuned := fixturePair(t)
	code, out, _ := run("diff", base, tuned)
	if code != ExitDifferent {
		t.Fatalf("changed diff exit = %d", code)
	}
	for _, want := range []string{
		"model.layers.1.attn.w", "changed layers (1 of", "verdict: DIFFERENT", "5.00e-01",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q:\n%s", want, out)
		}
	}

	// --quiet keeps the exit code but writes nothing.
	code, out, stderr := run("diff", "--quiet", base, tuned)
	if code != ExitDifferent || out != "" || stderr != "" {
		t.Fatalf("quiet mode: code=%d out=%q stderr=%q", code, out, stderr)
	}
}

func TestDiffJSONIsValidAndComplete(t *testing.T) {
	base, tuned := fixturePair(t)
	code, out, _ := run("diff", "--format", "json", base, tuned)
	if code != ExitDifferent {
		t.Fatalf("exit = %d", code)
	}
	var doc struct {
		Tool    string `json:"tool"`
		Summary struct {
			Compared  int `json:"compared"`
			Identical int `json:"identical"`
			Changed   int `json:"changed"`
		} `json:"summary"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if doc.Tool != "layerdiff" || doc.Summary.Compared != 3 || doc.Summary.Changed != 1 {
		t.Fatalf("JSON payload wrong: %+v", doc)
	}
}

func TestDiffMarkdownFormat(t *testing.T) {
	base, tuned := fixturePair(t)
	code, out, _ := run("diff", "--format", "markdown", base, tuned)
	if code != ExitDifferent {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(out, "**Verdict: DIFFERENT**") || !strings.Contains(out, "| compared |") {
		t.Fatalf("markdown output wrong:\n%s", out)
	}
}

func TestDiffToleranceFlagsFlipVerdict(t *testing.T) {
	base, tuned := fixturePair(t)
	code, out, _ := run("diff", "--atol", "1", base, tuned)
	if code != ExitSame {
		t.Fatalf("atol=1 should absorb the 0.5 delta, exit = %d", code)
	}
	if !strings.Contains(out, "EQUIVALENT") {
		t.Fatalf("verdict should be EQUIVALENT:\n%s", out)
	}
}

func TestDiffUsageErrors(t *testing.T) {
	base, tuned := fixturePair(t)
	cases := [][]string{
		{"diff", base}, // one path
		{"diff", "--format", "yaml", base, tuned},  // bad format
		{"diff", "--atol", "-1", base, tuned},      // negative tolerance
		{"diff", "--include", "[bad", base, tuned}, // malformed pattern
		{"ls", "--format", "xml", base},            // bad ls format
		{"ls"},                                     // missing path
		{"hash"},                                   // missing path
	}
	for _, args := range cases {
		if code, _, _ := run(args...); code != ExitUsage {
			t.Errorf("%v: exit = %d, want %d", args, code, ExitUsage)
		}
	}
}

func TestDiffMissingFileIsRuntimeError(t *testing.T) {
	base, _ := fixturePair(t)
	code, _, stderr := run("diff", base, filepath.Join(t.TempDir(), "missing.safetensors"))
	if code != ExitRuntime || stderr == "" {
		t.Fatalf("missing file: code=%d stderr=%q", code, stderr)
	}
}

func TestDiffIncludeExcludeFilters(t *testing.T) {
	base, tuned := fixturePair(t)
	// Excluding the only changed tensor turns the diff identical.
	code, _, _ := run("diff", "--exclude", "model.layers.1.*", base, tuned)
	if code != ExitSame {
		t.Fatalf("exclude should hide the change, exit = %d", code)
	}
	code, _, _ = run("diff", "--include", "model.norm.*", base, tuned)
	if code != ExitSame {
		t.Fatalf("include of unchanged tensor should be identical, exit = %d", code)
	}
}

func TestDiffHashOnlyStillDetectsChange(t *testing.T) {
	base, tuned := fixturePair(t)
	code, out, _ := run("diff", "--hash-only", base, tuned)
	if code != ExitDifferent {
		t.Fatalf("hash-only exit = %d", code)
	}
	if !strings.Contains(out, "mode: hash-only") {
		t.Fatalf("hash-only banner missing:\n%s", out)
	}
}

func TestHashThenDiffManifestAgainstCheckpoint(t *testing.T) {
	base, tuned := fixturePair(t)
	man := filepath.Join(t.TempDir(), "base.manifest.json")
	code, _, stderr := run("hash", "-o", man, base)
	if code != ExitSame {
		t.Fatalf("hash: code=%d stderr=%q", code, stderr)
	}

	// Manifest vs the same checkpoint: identical, exit 0.
	if code, _, _ := run("diff", man, base); code != ExitSame {
		t.Fatalf("manifest vs same checkpoint: exit = %d", code)
	}
	// Manifest vs the tuned checkpoint: hash-only diff, exit 1.
	code, out, _ := run("diff", man, tuned)
	if code != ExitDifferent {
		t.Fatalf("manifest vs tuned: exit = %d", code)
	}
	if !strings.Contains(out, "mode: hash-only") || !strings.Contains(out, "model.layers.1.attn.w") {
		t.Fatalf("manifest diff output wrong:\n%s", out)
	}

	// Without -o the manifest goes to stdout and must be valid JSON.
	code, out, _ = run("hash", base)
	if code != ExitSame {
		t.Fatalf("hash to stdout exit = %d", code)
	}
	var doc struct {
		Marker  int            `json:"layerdiff_manifest"`
		Tensors map[string]any `json:"tensors"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("manifest not valid JSON: %v", err)
	}
	if doc.Marker != 1 || len(doc.Tensors) != 3 {
		t.Fatalf("manifest wrong: %+v", doc)
	}
}

func TestLsTextAndJSON(t *testing.T) {
	base, _ := fixturePair(t)
	code, out, _ := run("ls", base)
	if code != ExitSame {
		t.Fatalf("ls exit = %d", code)
	}
	if !strings.Contains(out, "3 tensors") || !strings.Contains(out, "model.norm.weight") {
		t.Fatalf("ls output wrong:\n%s", out)
	}

	code, out, _ = run("ls", "--format", "json", "--hash", base)
	if code != ExitSame {
		t.Fatalf("ls json exit = %d", code)
	}
	var doc struct {
		Tensors []struct {
			Name   string `json:"name"`
			SHA256 string `json:"sha256"`
		} `json:"tensors"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("ls JSON invalid: %v", err)
	}
	if len(doc.Tensors) != 3 || len(doc.Tensors[0].SHA256) != 64 {
		t.Fatalf("ls JSON wrong: %+v", doc)
	}

	// --stats adds streamed numeric columns to the text view.
	code, out, _ = run("ls", "--stats", base)
	if code != ExitSame {
		t.Fatalf("ls --stats exit = %d", code)
	}
	if !strings.Contains(out, "mean") || !strings.Contains(out, "rms") {
		t.Fatalf("stats columns missing:\n%s", out)
	}
}

func TestDiffShardedIndexEqualsSingleFile(t *testing.T) {
	// The same logical model stored single-file vs sharded must diff as
	// identical: layout is irrelevant, content identity is what counts.
	base, _ := fixturePair(t)
	dir := t.TempDir()
	if err := stwrite.WriteFile(filepath.Join(dir, "model-00001-of-00002.safetensors"), nil, []stwrite.Tensor{
		{Name: "model.layers.0.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(1, 2, 3, 4)},
	}); err != nil {
		t.Fatal(err)
	}
	if err := stwrite.WriteFile(filepath.Join(dir, "model-00002-of-00002.safetensors"), nil, []stwrite.Tensor{
		{Name: "model.layers.1.attn.w", DType: "F32", Shape: []int64{2, 2}, Data: stwrite.F32(5, 6, 7, 8)},
		{Name: "model.norm.weight", DType: "F32", Shape: []int64{2}, Data: stwrite.F32(1, 1)},
	}); err != nil {
		t.Fatal(err)
	}
	index := map[string]any{"weight_map": map[string]string{
		"model.layers.0.attn.w": "model-00001-of-00002.safetensors",
		"model.layers.1.attn.w": "model-00002-of-00002.safetensors",
		"model.norm.weight":     "model-00002-of-00002.safetensors",
	}}
	js, _ := json.Marshal(index)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), js, 0o644); err != nil {
		t.Fatal(err)
	}

	code, out, stderr := run("diff", base, dir)
	if code != ExitSame {
		t.Fatalf("sharded == single-file: code=%d stderr=%q\n%s", code, stderr, out)
	}
}

func TestDiffTopFlagCapsTable(t *testing.T) {
	dir := t.TempDir()
	var a, b []stwrite.Tensor
	for _, name := range []string{"l.0.w", "l.1.w", "l.2.w"} {
		a = append(a, stwrite.Tensor{Name: name, DType: "F32", Shape: []int64{1}, Data: stwrite.F32(1)})
		b = append(b, stwrite.Tensor{Name: name, DType: "F32", Shape: []int64{1}, Data: stwrite.F32(2)})
	}
	pa, pb := filepath.Join(dir, "a.safetensors"), filepath.Join(dir, "b.safetensors")
	if err := stwrite.WriteFile(pa, nil, a); err != nil {
		t.Fatal(err)
	}
	if err := stwrite.WriteFile(pb, nil, b); err != nil {
		t.Fatal(err)
	}
	_, out, _ := run("diff", "--top", "2", pa, pb)
	if !strings.Contains(out, "(2 of 3 shown") {
		t.Fatalf("--top not applied:\n%s", out)
	}
}
