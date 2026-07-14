// The hash subcommand: snapshot a checkpoint's identity as a manifest —
// every tensor's dtype, shape, size, and SHA-256 — so future diffs don't
// need the original weights.
package cli

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/JaydenCJ/layerdiff/internal/checkpoint"
	"github.com/JaydenCJ/layerdiff/internal/manifest"
)

func runHash(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("hash", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	out := fs.String("o", "", "write the manifest to this file instead of stdout")
	if err := fs.Parse(args); err != nil {
		return parseErr(err, "hash", stdout, stderr)
	}
	if fs.NArg() != 1 {
		return usageErr(stderr, "hash: want exactly one path, got %d", fs.NArg())
	}

	cp, err := checkpoint.Open(fs.Arg(0))
	if err != nil {
		return runtimeErr(stderr, err)
	}
	m, err := manifest.Build(cp)
	if err != nil {
		return runtimeErr(stderr, err)
	}

	if *out == "" {
		if err := m.Write(stdout); err != nil {
			return runtimeErr(stderr, err)
		}
		return ExitSame
	}
	f, err := os.Create(*out)
	if err != nil {
		return runtimeErr(stderr, err)
	}
	if err := m.Write(f); err != nil {
		f.Close()
		return runtimeErr(stderr, err)
	}
	// Close errors are real write failures (e.g. a full disk flushing the
	// last block); a manifest is an audit artifact, so never report
	// success unless every byte landed.
	if err := f.Close(); err != nil {
		return runtimeErr(stderr, err)
	}
	noun := "tensors"
	if cp.Len() == 1 {
		noun = "tensor"
	}
	fmt.Fprintf(stdout, "wrote manifest for %d %s to %s\n", cp.Len(), noun, *out)
	return ExitSame
}
