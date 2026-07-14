# layerdiff examples

Two runnable pieces, both offline and deterministic.

## make-demo

A tiny Go program that fabricates a realistic checkpoint pair without
any ML framework: `base/` is a classic single-file `.safetensors`
model, `tuned/` is the same model sharded into two files plus a
`model.safetensors.index.json` weight map — as produced for large
models. "Finetuning" perturbed layer 2's attention and MLP projections,
nudged layer 1's norm by ~1e-3, dropped a rotary buffer, and added a
gate projection. Values come from a fixed xorshift PRNG, so every run
writes byte-identical files.

```bash
go run ./examples/make-demo /tmp/layerdiff-demo
layerdiff diff /tmp/layerdiff-demo/base /tmp/layerdiff-demo/tuned
layerdiff diff --atol 0.002 /tmp/layerdiff-demo/base /tmp/layerdiff-demo/tuned
```

The second command shows tolerance at work: the 1e-3 norm nudge drops
from `changed` to `equivalent` and only the real finetuning targets
remain.

## audit-gate.sh

Shows `layerdiff diff --quiet` as a release gate: it exits 1 the moment
two checkpoints differ, so a publish script can refuse to ship weights
that silently drifted from what was evaluated. It also demonstrates the
manifest workflow — hash the evaluated checkpoint once, keep only the
small JSON, and audit against it later without the original weights.

```bash
bash examples/audit-gate.sh /tmp/layerdiff-demo/base /tmp/layerdiff-demo/tuned; echo "exit: $?"
```

Both examples pin every value, so their output is identical on every
machine.
