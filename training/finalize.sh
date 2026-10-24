#!/usr/bin/env bash
# Build a servable student from a recipe's 5-fold run and its deployed model.
#   ./training/finalize.sh runs/folds_st runs/full_st exports/student
# 1. structural calibration (label-noise floor, per-queue bias, approve ceiling)
#    from the 5-fold out-of-fold predictions (6,500 rows)
# 2. scale refit + abstain thresholds on the deployed model's dev predictions,
#    with a cross-fitted dev estimate printed for honesty
# 3. ONNX export, then latency + parity on 4 cores
set -euo pipefail
FOLDS=$1; DEPLOY=$2; OUT=$3
P=${PY:-python}
$P training/eval_runs.py "$FOLDS" --save_policy "$DEPLOY/struct_policy.json" | grep -E "abstain|TOTAL" || true
$P training/fit_policy_dev.py "$DEPLOY/dev_logits.npz" --struct "$DEPLOY/struct_policy.json" --out "$DEPLOY/policy.json" \
  | grep -E "cross-fitted|TOTAL|route|approve|ground" || true
$P training/export_onnx.py "$DEPLOY" "$OUT" --policy "$DEPLOY/policy.json" > /dev/null 2>&1
cp "$DEPLOY/dev_logits.npz" "$OUT/"
[ -f "$DEPLOY/approve_stage.json" ] && cp "$DEPLOY/approve_stage.json" "$OUT/"
taskset -c 0-3 $P tools/bench_cpu.py "$OUT" --n 300 --threads 4 --variants model.onnx | grep -v -i warn
