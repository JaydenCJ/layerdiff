# layerdiff diff format

This document pins down what a layerdiff comparison means: the tensor
classes, the numeric definitions, the layer-grouping rule, and the JSON
layout (`schema_version: 1`). The JSON schema only changes with a major
version bump of `schema_version`.

## Tensor classes

Every tensor in the union of both checkpoints lands in exactly one
class:

| Class | Meaning |
|---|---|
| `identical` | Byte-for-byte equal: same SHA-256 over the raw data. Counted in the summary, not listed. |
| `equivalent` | Bytes differ, but no element is outside the tolerance. Without tolerance flags this still catches `-0.0` vs `0.0` and differing NaN payloads. |
| `changed` | At least one element differs beyond `atol + rtol×|new|`. |
| `added` / `removed` | Present only in the second / first checkpoint. |
| `dtype-mismatch` / `shape-mismatch` | Present in both but not elementwise comparable. Both sides are still hashed and characterized individually. |

The process exit code is 1 exactly when
`changed + added + removed + mismatched > 0`; an outcome that is all
`identical`/`equivalent` exits 0.

## Elementwise semantics

- An element pair counts as changed when `|b−a| > atol + rtol×|b|`
  (numpy's `isclose` convention, with the second checkpoint as the
  reference). Defaults are `atol=0 rtol=0`, i.e. exact equality.
- **NaN**: NaN facing NaN is agreement (regardless of payload bits);
  NaN facing a number is a change no tolerance can excuse. NaN pairs
  never enter the numeric aggregates.
- **Infinities**: same-signed infinities agree; anything else involving
  an infinity is an infinite difference.
- Difference metrics per changed tensor: `max_abs` (+ flat index),
  `mean_abs`, `l1`, `l2`, `rel_l2` = ‖b−a‖₂/‖a‖₂, and `cosine`
  similarity of the two tensors as flat vectors.
- Per-side statistics: element count, zeros, NaNs, Infs, min, max,
  mean, RMS, L2 — all computed over non-NaN elements in one streamed
  pass.

Non-finite numbers cannot appear in JSON, so they are encoded as the
strings `"NaN"`, `"Infinity"`, and `"-Infinity"`.

## Layer grouping

The layer key of a tensor name is, by default, the prefix up to and
including the **first all-digit segment**:
`model.layers.17.attn.wq.weight → model.layers.17`. Names without a
digit segment drop their final segment (the parameter name):
`model.embed_tokens.weight → model.embed_tokens`. Override with
`--group-depth N` to use the first N dot-segments instead.

Layer rows aggregate their tensors: `max_abs` is the maximum, `mean_abs`
is Σ|Δ| divided by the layer's numeric element pairs, and `rel_l2` is
the layer-wide √(Σ‖Δ‖²)/√(Σ‖a‖²). Only layers containing at least one
non-identical, non-equivalent tensor are listed.

## JSON layout

```text
{
  "tool": "layerdiff", "version": "0.1.0", "schema_version": 1,
  "path_a": …, "path_b": …, "kind_a": …, "kind_b": …,
  "hash_only": false, "atol": 0, "rtol": 0,
  "summary": { tensors_a, tensors_b, compared, identical, equivalent,
               changed, added, removed, mismatched,
               layers, layers_changed, bytes_a, bytes_b },
  "layers":  [ { layer, tensors, identical, equivalent, changed, added,
                 removed, mismatched, max_abs, mean_abs, rel_l2 }, … ],
  "tensors": [ { name, layer, class, dtype_a, dtype_b, shape_a, shape_b,
                 bytes_a, bytes_b, sha256_a, sha256_b,
                 stats_a, stats_b, metrics }, … ]
}
```

`tensors` lists every non-identical tensor in natural name order
(numeric segments compared as numbers), never truncated — `--top` only
limits the terminal tables. Identical inputs yield byte-identical
output across runs and machines.

## Hash manifests

`layerdiff hash` emits `{"layerdiff_manifest": 1, "tool", "version",
"source", "tensors": {name: {dtype, shape, bytes, sha256}}}`. A manifest
can replace either side of `diff`; the comparison then degrades to
identity checking (`hash_only: true`, no elementwise metrics), which is
exactly enough for "did anything change since the run I archived?".

## Streaming guarantees

Tensor data is read in fixed 256 KiB chunks per side; memory use is
constant regardless of checkpoint size. Each considered tensor is read
exactly once per side, even when hashes, statistics, and difference
metrics are all requested.
