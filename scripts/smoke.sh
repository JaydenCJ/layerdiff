#!/usr/bin/env bash
# End-to-end smoke test for layerdiff: builds the binary, fabricates a
# deterministic base/tuned checkpoint pair (single-file vs sharded), and
# asserts on real CLI output and exit codes. No network, idempotent,
# finishes in seconds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

fail() {
  echo "SMOKE FAIL: $*" >&2
  exit 1
}

BIN="$WORKDIR/layerdiff"
DEMO="$WORKDIR/demo"

echo "1. build"
(cd "$ROOT" && go build -o "$BIN" ./cmd/layerdiff) || fail "go build failed"

echo "2. version matches manifest"
"$BIN" --version | grep -qx "layerdiff 0.1.0" || fail "--version mismatch"

echo "3. fabricate base (single file) and tuned (sharded) checkpoints"
(cd "$ROOT" && go run ./examples/make-demo "$DEMO" >/dev/null) || fail "make-demo failed"
[ -f "$DEMO/base/model.safetensors" ] || fail "base checkpoint missing"
[ -f "$DEMO/tuned/model.safetensors.index.json" ] || fail "tuned index missing"

echo "4. identical inputs exit 0 with verdict IDENTICAL"
OUT="$("$BIN" diff "$DEMO/base" "$DEMO/base")" || fail "self-diff must exit 0"
echo "$OUT" | grep -q "verdict: IDENTICAL" || fail "self-diff verdict wrong"

echo "5. base vs tuned exits 1 and names the touched layers"
set +e
OUT="$("$BIN" diff "$DEMO/base" "$DEMO/tuned")"
CODE=$?
set -e
[ "$CODE" -eq 1 ] || fail "diff should exit 1, got $CODE"
echo "$OUT" | grep -q "5 changed" || fail "changed count wrong"
echo "$OUT" | grep -q "1 added" || fail "added count wrong"
echo "$OUT" | grep -q "1 removed" || fail "removed count wrong"
echo "$OUT" | grep -q "changed layers (4 of 7)" || fail "layer rollup wrong"
echo "$OUT" | grep -q "model.layers.2.attn.wq.weight" || fail "top changed tensor missing"
echo "$OUT" | grep -q "verdict: DIFFERENT" || fail "verdict wrong"

echo "6. JSON report is machine-readable and complete"
set +e
JSON="$("$BIN" diff --format json "$DEMO/base" "$DEMO/tuned")"
set -e
echo "$JSON" | grep -q '"tool": "layerdiff"' || fail "json envelope missing"
echo "$JSON" | grep -q '"schema_version": 1' || fail "schema version missing"
echo "$JSON" | grep -q '"changed": 5' || fail "json changed count wrong"
echo "$JSON" | grep -q '"class": "added"' || fail "json class missing"

echo "7. tolerance reclassifies a tiny norm nudge as equivalent"
set +e
OUT="$("$BIN" diff --atol 0.002 "$DEMO/base" "$DEMO/tuned")"
set -e
echo "$OUT" | grep -q "1 equivalent" || fail "tolerance not applied"
echo "$OUT" | grep -q "4 changed" || fail "tolerance changed count wrong"

echo "8. hash manifest replaces the original checkpoint in a diff"
"$BIN" hash -o "$WORKDIR/base.manifest.json" "$DEMO/base" >/dev/null || fail "hash failed"
"$BIN" diff --quiet "$WORKDIR/base.manifest.json" "$DEMO/base" \
  || fail "manifest vs same checkpoint should exit 0"
set +e
OUT="$("$BIN" diff "$WORKDIR/base.manifest.json" "$DEMO/tuned")"
CODE=$?
set -e
[ "$CODE" -eq 1 ] || fail "manifest vs tuned should exit 1, got $CODE"
echo "$OUT" | grep -q "mode: hash-only" || fail "hash-only banner missing"

echo "9. exclude filters can carve the finetuned region out"
"$BIN" diff --quiet \
  --exclude 'model.layers.2.*' --exclude 'model.layers.1.norm.*' \
  --exclude 'lm_head.*' --exclude 'model.rotary.*' \
  "$DEMO/base" "$DEMO/tuned" || fail "excluding all touched tensors should exit 0"

echo "10. ls inventories the sharded checkpoint"
"$BIN" ls "$DEMO/tuned" | grep -q "22 tensors" || fail "ls tensor count wrong"
"$BIN" ls "$DEMO/tuned" | grep -q "sharded checkpoint, 2 shards" || fail "ls kind wrong"

echo "11. usage errors exit 2, runtime errors exit 3"
set +e
"$BIN" diff --format yaml "$DEMO/base" "$DEMO/tuned" >/dev/null 2>&1
[ $? -eq 2 ] || fail "bad --format should exit 2"
"$BIN" diff "$DEMO/base" "$WORKDIR/missing.safetensors" >/dev/null 2>&1
[ $? -eq 3 ] || fail "missing checkpoint should exit 3"
set -e

echo "SMOKE OK"
