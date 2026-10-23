#!/usr/bin/env bash
# Finalize candidate runs on the GPU box, pull them to this machine and stress-test each.
#   tools/eval_candidates.sh <folds_run> <full_run> [<full_run> ...]
# e.g.  tools/eval_candidates.sh runs/v6g1_e17 runs/v6g1h_e17_s0 runs/v6g1h_e17_s3
# Results land in docs/stress/<name>.json; compare with tools/compare_stress.py docs/stress/*.json
set -euo pipefail
HOST=${HOST:-sra@10.27.100.101}
REMOTE=${REMOTE:-sahil/decide}   # relative to the remote home
CACHE=${CACHE:-${TMPDIR:-/tmp}/decide_candidates}
FOLDS=$1; shift
mkdir -p docs/stress "$CACHE"
for RUN in "$@"; do
  NAME=$(basename "$RUN" | sed 's/_e17//')
  echo "=== $NAME"
  ssh "$HOST" "cd $REMOTE && PY=.venv/bin/python ./finalize.sh $FOLDS $RUN exports/$NAME 2>&1 | grep -E 'cross-fitted|mean p95' | grep -v -i warn"
  mkdir -p "$CACHE/$NAME/student"
  for F in meta.json model.onnx policy.json tokenizer.json; do
    scp -q "$HOST:$REMOTE/exports/$NAME/$F" "$CACHE/$NAME/student/"
  done
  .venv/bin/python tools/stress_test.py --artifacts "$CACHE/$NAME" --out "docs/stress/$NAME.json" 2>&1 \
    | grep -v -i warn | grep -E "^(clean|all_mixed|draft_ticket_ref|num_words)"
done
