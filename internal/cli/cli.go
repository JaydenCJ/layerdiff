// Package cli implements the layerdiff command-line interface. Run takes
// argv and two writers and returns an exit code, so the whole surface is
// testable in-process without building a binary.
package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/JaydenCJ/layerdiff/internal/version"
)

// Exit codes, documented in the README. `diff` mirrors POSIX diff:
// 0 means no differences, 1 means differences were found.
const (
	ExitSame      = 0
	ExitDifferent = 1
	ExitUsage     = 2
	ExitRuntime   = 3
)

// Run dispatches argv and returns the process exit code.
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return ExitUsage
	}
	switch args[0] {
	case "diff":
		return runDiff(args[1:], stdout, stderr)
	case "ls":
		return runLs(args[1:], stdout, stderr)
	case "hash":
		return runHash(args[1:], stdout, stderr)
	case "version", "--version", "-v":
		fmt.Fprintf(stdout, "layerdiff %s\n", version.Version)
		return ExitSame
	case "help", "--help", "-h":
		usage(stdout)
		return ExitSame
	default:
		if strings.HasPrefix(args[0], "-") {
			fmt.Fprintf(stderr, "layerdiff: unknown flag %q before a subcommand\n\n", args[0])
		} else {
			fmt.Fprintf(stderr, "layerdiff: unknown command %q\n\n", args[0])
		}
		usage(stderr)
		return ExitUsage
	}
}

func usage(w io.Writer) {
	fmt.Fprint(w, `layerdiff — compare two model checkpoints tensor by tensor

Usage:
  layerdiff diff [flags] A B     compare checkpoints (or hash manifests)
  layerdiff ls   [flags] PATH    list one checkpoint's tensors
  layerdiff hash [flags] PATH    write a hash manifest for later diffing
  layerdiff version              print the version

A checkpoint path is a .safetensors file, a *.safetensors.index.json
sharded index, or a directory containing either. A .json file that was
produced by "layerdiff hash" is compared by tensor identity alone.

Diff flags:
  --format text|json|markdown   output format (default text)
  --atol N                      absolute tolerance per element (default 0)
  --rtol N                      relative tolerance per element (default 0)
  --include GLOB                only compare matching tensor names (repeatable)
  --exclude GLOB                skip matching tensor names (repeatable)
  --group-depth N               layer key = first N name segments (default auto)
  --top N                       changed-tensor rows in text/markdown (default 20, 0=all)
  --hash-only                   skip statistics; compare tensor digests only
  --quiet                       no output, communicate via exit code

Ls flags:
  --format text|json            output format (default text)
  --hash                        stream each tensor and include its SHA-256
  --stats                       stream each tensor and include numeric statistics
  --include / --exclude GLOB    same name filters as diff (repeatable)

Hash flags:
  -o FILE                       write the manifest to FILE instead of stdout

Exit codes: 0 no differences, 1 differences found, 2 usage error,
3 runtime error (unreadable or malformed checkpoint).
`)
}

// parseErr handles a flag.FlagSet parse failure uniformly: --help/-h on
// a subcommand prints the usage to stdout and exits 0; anything else is
// a usage error.
func parseErr(err error, sub string, stdout, stderr io.Writer) int {
	if errors.Is(err, flag.ErrHelp) {
		usage(stdout)
		return ExitSame
	}
	return usageErr(stderr, "%s: %v", sub, err)
}

// multiFlag is a repeatable string flag.
type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

// usageErr prints a message plus context line and returns ExitUsage.
func usageErr(stderr io.Writer, format string, args ...any) int {
	fmt.Fprintf(stderr, "layerdiff: "+format+"\n", args...)
	fmt.Fprintln(stderr, `run "layerdiff help" for usage`)
	return ExitUsage
}

// runtimeErr prints a message and returns ExitRuntime.
func runtimeErr(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "layerdiff: %v\n", err)
	return ExitRuntime
}
