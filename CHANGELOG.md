# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-12

### Added

- safetensors header parsing with hard validation: offset bounds,
  shape/byte-length consistency, negative-dimension and overflow
  rejection, a 100 MiB header cap, and graceful "opaque" handling of
  dtypes newer than this build.
- Checkpoint resolution for all three real-world layouts: a single
  `.safetensors` file, a `*.safetensors.index.json` sharded weight map
  (with escape-proof shard-name validation), and a directory of loose
  shards merged with duplicate detection.
- Streamed, constant-memory scanning (256 KiB chunks): per-tensor
  SHA-256 plus min/max/mean/RMS/L2 and zero/NaN/Inf counts, with exact
  decoders for F64/F32/F16/BF16, all integer widths, and BOOL.
- Lockstep pair scanning producing elementwise difference metrics —
  max|Δ| (+ index), mean|Δ|, L1/L2 norms, relative L2, cosine
  similarity, changed-element counts — with careful NaN, ±Inf, and
  negative-zero semantics, in one read per side.
- `diff` subcommand with identical/equivalent/changed/added/removed/
  mismatch classification, `--atol`/`--rtol` tolerances, per-layer
  rollup (auto layer keys or `--group-depth`), `--include`/`--exclude`
  name globs, `--top`, `--quiet`, and POSIX-diff exit codes.
- Text, JSON (`schema_version: 1`, non-finite-safe encoding), and
  Markdown renderers with fully deterministic, naturally sorted output.
- `ls` subcommand inventorying one checkpoint (dtype, shape, bytes,
  optional `--hash` / `--stats`) and `hash` subcommand emitting a
  manifest that can replace either side of a future diff.
- Runnable examples (`examples/make-demo`, `examples/audit-gate.sh`)
  and a format reference (`docs/diff-format.md`).
- 90 deterministic offline tests (unit + in-process CLI integration
  against fabricated checkpoints) and `scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/layerdiff/releases/tag/v0.1.0
