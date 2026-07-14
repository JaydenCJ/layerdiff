#!/usr/bin/env bash
# audit-gate.sh EVALUATED CANDIDATE
#
# Refuses (exit 1) when CANDIDATE's tensors differ from the EVALUATED
# checkpoint — the guard you want in front of "upload the weights".
# Also shows the manifest workflow: snapshot the evaluated checkpoint's
# identity once, then audit later runs against the small JSON instead of
# keeping the original weights around.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: audit-gate.sh EVALUATED CANDIDATE" >&2
  exit 2
fi
EVALUATED="$1"
CANDIDATE="$2"

LAYERDIFF="${LAYERDIFF:-layerdiff}"
SNAPSHOT="$(mktemp -t layerdiff-manifest.XXXXXX.json)"
trap 'rm -f "$SNAPSHOT"' EXIT

# 1. Snapshot the evaluated checkpoint: dtype/shape/SHA-256 per tensor.
"$LAYERDIFF" hash -o "$SNAPSHOT" "$EVALUATED"

# 2. Gate the candidate against the snapshot. --quiet: exit code only.
if "$LAYERDIFF" diff --quiet "$SNAPSHOT" "$CANDIDATE"; then
  echo "gate: PASS — candidate is tensor-identical to the evaluated checkpoint"
  exit 0
fi

echo "gate: FAIL — candidate diverged; changed layers:"
"$LAYERDIFF" diff --top 5 "$SNAPSHOT" "$CANDIDATE" || true
exit 1
