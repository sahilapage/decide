#!/usr/bin/env bash
# Finalize candidate runs on the GPU box and stress-test each THERE (keeps the
# laptop cool), then pull the result files back.
#   tools/eval_candidates.sh <folds_run> <full_run> [<full_run> ...]
# e.g.  tools/eval_candidates.sh runs/v6g1_e17 runs/v6g1h_e17_s0 runs/v6g1h_e17_s3
# Results land in docs/stress/<name>.json; compare with tools/compare_stress.py docs/stress/*.json
# STRESS_LOCAL=1 runs the stress test on this machine instead.
set -euo pipefail
: "${HOST:?set HOST=user@gpu-box}"
REMOTE=${REMOTE:-decide}        # repo path relative to the remote home
CACHE=${CACHE:-${TMPDIR:-/tmp}/decide_candidates}
CPUS=${CPUS:-8-11}               # remote cores for the stress test
FOLDS=$1; shift
mkdir -p docs/stress "$CACHE"
# serving code must match on both sides
scp -q decide/*.py "$HOST:$REMOTE/decide/"
ssh "$HOST" "mkdir -p $REMOTE/training"; scp -q model.py "$HOST:$REMOTE/"; scp -q training/finalize.sh training/pack_student.py "$HOST:$REMOTE/training/"
scp -q tools/stress_test.py tools/judge_cases.py "$HOST:$REMOTE/tools/"
ssh "$HOST" "mkdir -p $REMOTE/tests"; scp -q tests/judge_cases.json tests/complex_cases.json "$HOST:$REMOTE/tests/"
for RUN in "$@"; do
  NAME=$(basename "$RUN" | sed 's/_e17//')
  echo "=== $NAME"
  ssh "$HOST" "cd $REMOTE && PY=.venv/bin/python ./training/finalize.sh $FOLDS $RUN exports/$NAME 2>&1 | grep -E 'cross-fitted|mean p95' | grep -v -i warn"
  if [ "${STRESS_LOCAL:-0}" = "1" ]; then
    mkdir -p "$CACHE/$NAME/student"
    for F in meta.json model.onnx policy.json tokenizer.json; do
      scp -q "$HOST:$REMOTE/exports/$NAME/$F" "$CACHE/$NAME/student/"
    done
    scp -q "$HOST:$REMOTE/exports/$NAME/approve_stage.json" "$CACHE/$NAME/student/" 2>/dev/null || true
    .venv/bin/python tools/stress_test.py --artifacts "$CACHE/$NAME" --out "docs/stress/$NAME.json" 2>&1 \
      | grep -v -i warn | grep -E "^(clean|all_mixed|draft_ticket_ref|kb_single)"
  else
    ssh "$HOST" "cd $REMOTE && mkdir -p cand/$NAME/student docs/stress && rm -rf cand/$NAME/student/* && cp exports/$NAME/meta.json exports/$NAME/model.onnx exports/$NAME/policy.json exports/$NAME/tokenizer.json cand/$NAME/student/ \
      && { [ ! -f exports/$NAME/approve_stage.json ] || cp exports/$NAME/approve_stage.json cand/$NAME/student/; } \
      && taskset -c $CPUS .venv/bin/python tools/stress_test.py --artifacts cand/$NAME --out docs/stress/$NAME.json 2>&1 \
      | grep -v -i warn | grep -E '^(clean|all_mixed|draft_ticket_ref|kb_single)'"
    scp -q "$HOST:$REMOTE/docs/stress/$NAME.json" docs/stress/
    ssh "$HOST" "cd $REMOTE && taskset -c $CPUS .venv/bin/python tools/judge_cases.py --artifacts cand/$NAME 2>&1 | grep -E 'XX|as expected'; taskset -c $CPUS .venv/bin/python tools/judge_cases.py --artifacts cand/$NAME --cases tests/complex_cases.json 2>&1 | grep -E 'XX|as expected'"
  fi
done
