// The diff subcommand: resolve both sides (checkpoint or manifest),
// run the comparison, render, and translate the outcome into an exit
// code that scripts can gate on.
package cli

import (
	"flag"
	"io"
	"strings"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/diff"
	"github.com/JaydenCJ/layerdiff/internal/manifest"
	"github.com/JaydenCJ/layerdiff/internal/render"
	"github.com/JaydenCJ/layerdiff/internal/scan"
)

func runDiff(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("diff", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var (
		format     = fs.String("format", "text", "output format: text, json, or markdown")
		atol       = fs.Float64("atol", 0, "absolute tolerance per element")
		rtol       = fs.Float64("rtol", 0, "relative tolerance per element")
		groupDepth = fs.Int("group-depth", 0, "layer key = first N name segments (0 = auto)")
		top        = fs.Int("top", 20, "changed-tensor rows in text/markdown output (0 = all)")
		hashOnly   = fs.Bool("hash-only", false, "compare tensor digests only, skip statistics")
		quiet      = fs.Bool("quiet", false, "no output, communicate via exit code")
		include    multiFlag
		exclude    multiFlag
	)
	fs.Var(&include, "include", "only compare matching tensor names (repeatable)")
	fs.Var(&exclude, "exclude", "skip matching tensor names (repeatable)")
	if err := fs.Parse(args); err != nil {
		return parseErr(err, "diff", stdout, stderr)
	}
	if fs.NArg() != 2 {
		return usageErr(stderr, "diff: want exactly two paths, got %d", fs.NArg())
	}
	switch *format {
	case "text", "json", "markdown":
	default:
		return usageErr(stderr, "diff: unknown --format %q (want text, json, or markdown)", *format)
	}
	if *atol < 0 || *rtol < 0 {
		return usageErr(stderr, "diff: tolerances must be non-negative")
	}
	filter, err := diff.NewFilter(include, exclude)
	if err != nil {
		return usageErr(stderr, "diff: %v", err)
	}

	pathA, pathB := fs.Arg(0), fs.Arg(1)
	opts := diff.Options{
		Tolerance:  scan.Tolerance{Atol: *atol, Rtol: *rtol},
		GroupDepth: *groupDepth,
		Filter:     filter,
	}

	rep, err := compare(pathA, pathB, opts, *hashOnly)
	if err != nil {
		return runtimeErr(stderr, err)
	}

	if !*quiet {
		switch *format {
		case "json":
			if err := render.JSON(stdout, rep); err != nil {
				return runtimeErr(stderr, err)
			}
		case "markdown":
			render.Markdown(stdout, rep, *top)
		default:
			render.Text(stdout, rep, *top)
		}
	}
	if rep.Different() {
		return ExitDifferent
	}
	return ExitSame
}

// side is one resolved input: either a live checkpoint or a manifest.
type side struct {
	cp *checkpoint.Checkpoint
	mf *manifest.Manifest
}

// resolveSide decides what a path is. A .json file that sniffs as a
// layerdiff manifest is loaded as one; everything else goes through the
// checkpoint resolver (which also handles sharded index .json files).
func resolveSide(path string) (side, error) {
	if strings.HasSuffix(path, ".json") && manifest.Sniff(path) {
		mf, err := manifest.Load(path)
		if err != nil {
			return side{}, err
		}
		return side{mf: mf}, nil
	}
	cp, err := checkpoint.Open(path)
	if err != nil {
		return side{}, err
	}
	return side{cp: cp}, nil
}

// compare runs the right comparison for the pair of inputs. As soon as
// either side is identity-only (a manifest, or --hash-only), both sides
// are reduced to manifests and compared by digest.
func compare(pathA, pathB string, opts diff.Options, hashOnly bool) (*diff.Report, error) {
	a, err := resolveSide(pathA)
	if err != nil {
		return nil, err
	}
	b, err := resolveSide(pathB)
	if err != nil {
		return nil, err
	}

	if a.mf == nil && b.mf == nil && !hashOnly {
		return diff.Checkpoints(a.cp, b.cp, opts)
	}
	ma, err := toManifest(a)
	if err != nil {
		return nil, err
	}
	mb, err := toManifest(b)
	if err != nil {
		return nil, err
	}
	return diff.Manifests(ma, mb, pathA, pathB, opts)
}

func toManifest(s side) (*manifest.Manifest, error) {
	if s.mf != nil {
		return s.mf, nil
	}
	return manifest.Build(s.cp)
}
