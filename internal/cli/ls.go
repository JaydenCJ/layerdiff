// The ls subcommand: inventory one checkpoint — names, dtypes, shapes,
// sizes, and optionally streamed hashes and statistics.
package cli

import (
	"flag"
	"io"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/diff"
	"github.com/JaydenCJ/layerdiff/internal/render"
	"github.com/JaydenCJ/layerdiff/internal/scan"
)

func runLs(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ls", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var (
		format    = fs.String("format", "text", "output format: text or json")
		withHash  = fs.Bool("hash", false, "stream each tensor and include its SHA-256")
		withStats = fs.Bool("stats", false, "stream each tensor and include numeric statistics")
		include   multiFlag
		exclude   multiFlag
	)
	fs.Var(&include, "include", "only list matching tensor names (repeatable)")
	fs.Var(&exclude, "exclude", "skip matching tensor names (repeatable)")
	if err := fs.Parse(args); err != nil {
		return parseErr(err, "ls", stdout, stderr)
	}
	if fs.NArg() != 1 {
		return usageErr(stderr, "ls: want exactly one path, got %d", fs.NArg())
	}
	if *format != "text" && *format != "json" {
		return usageErr(stderr, "ls: unknown --format %q (want text or json)", *format)
	}
	filter, err := diff.NewFilter(include, exclude)
	if err != nil {
		return usageErr(stderr, "ls: %v", err)
	}

	cp, err := checkpoint.Open(fs.Arg(0))
	if err != nil {
		return runtimeErr(stderr, err)
	}

	listing := &render.Listing{
		Path:     cp.Path,
		Kind:     cp.Kind,
		Shards:   cp.Shards,
		Metadata: cp.Metadata,
	}
	for _, name := range cp.Names() {
		if !filter.Match(name) {
			continue
		}
		t, _ := cp.Tensor(name)
		entry := render.ListEntry{
			Name:  t.Name,
			DType: t.DType,
			Shape: t.Shape,
			Elems: t.Elems,
			Bytes: t.Bytes,
		}
		switch {
		case *withStats:
			res, err := scan.Tensor(t)
			if err != nil {
				return runtimeErr(stderr, err)
			}
			entry.SHA256, entry.Stats = res.SHA256, res.Stats
		case *withHash:
			res, err := scan.HashOnly(t)
			if err != nil {
				return runtimeErr(stderr, err)
			}
			entry.SHA256 = res.SHA256
		}
		listing.TotalBytes += t.Bytes
		listing.Tensors = append(listing.Tensors, entry)
	}
	if listing.Tensors == nil {
		listing.Tensors = []render.ListEntry{}
	}

	if *format == "json" {
		if err := render.ListingJSON(stdout, listing); err != nil {
			return runtimeErr(stderr, err)
		}
		return ExitSame
	}
	render.ListingText(stdout, listing, *withHash || *withStats, *withStats)
	return ExitSame
}
