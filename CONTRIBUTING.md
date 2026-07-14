# Contributing to layerdiff

Issues, discussions and pull requests are all welcome.

## Getting started

You need Go ≥1.22; nothing else — no ML framework, no model downloads.

```bash
git clone https://github.com/JaydenCJ/layerdiff && cd layerdiff
go build ./...
go test ./...
bash scripts/smoke.sh
```

`scripts/smoke.sh` builds the binary, fabricates a deterministic
base/tuned checkpoint pair (single-file and sharded) in a temp dir, and
asserts on real CLI output and exit codes across every subcommand; it
must finish by printing `SMOKE OK`.

## Before you open a pull request

1. `gofmt -l .` reports nothing (formatting is enforced).
2. `go vet ./...` passes with no findings.
3. `go test ./...` passes (90 deterministic tests, no network).
4. `bash scripts/smoke.sh` prints `SMOKE OK`.
5. Add tests for behavior changes; keep logic in pure, unit-testable
   modules (parsing, scanning, and classification never touch the
   terminal — only `internal/render` and `internal/cli` do).

## Ground rules

- Keep dependencies at zero; adding one needs strong justification in
  the PR. layerdiff's whole pitch is "no framework required".
- No network calls, ever. layerdiff only reads the files it is given.
  No telemetry.
- Determinism first: identical inputs must produce byte-identical
  reports, including all orderings and float formatting.
- Constant memory is a contract: tensor data may only be touched
  through the streaming scanners; never materialize a whole tensor.
- New dtypes are data: add a row in `internal/dtype/dtype.go` with an
  exact decoder, tests pinning the bit patterns (specials included),
  and a row in the README dtype table.
- Code comments and doc comments are written in English.

## Reporting bugs

Include the output of `layerdiff version`, the full command you ran,
and — for parsing problems — the first 16 bytes of the file plus its
header JSON if you can share it (`layerdiff ls PATH --format json`
output is ideal, it contains no tensor data). For misclassifications,
the two tensors' dtype/shape and the reported metrics are exactly what
the classifier sees.

## Security

Please do not open public issues for security problems; use GitHub's
private vulnerability reporting on this repository instead.
