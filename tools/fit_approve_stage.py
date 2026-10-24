"""
Fit the approve second stage on a recipe's out-of-fold logits and prepare
run directories that the usual pipeline (finalize.sh) can consume unchanged:

  runs/<folds>_st/   fold*.npz with val_approve replaced by the stage's 5-fold
                     cross-predicted logits (for the structural calibration)
  runs/<full>_st/    the full run with dev_logits' approve replaced by the
                     stage applied to the shipped network's dev logits, plus
                     approve_stage.json (model.pt etc. are symlinked)

  python tools/fit_approve_stage.py runs/v6t2_e17 runs/v6t2h_e17_s3
  ./training/finalize.sh runs/v6t2_e17_st runs/v6t2h_e17_s3_st exports/v6t2h_s3_st
"""

import json
import os
import shutil
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.approve_stage import ApproveStage, estimate_caps, features, fit_stage, make_model  # noqa: E402


def main():
    folds_dir, full_dir = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    tr = [json.loads(l) for l in open("data/train.jsonl")]
    dv = [json.loads(l) for l in open("data/dev.jsonl")]
    n = len(tr)
    oof = np.zeros(n)
    folds = [dict(np.load(os.path.join(folds_dir, "fold%d.npz" % f))) for f in range(5)]
    for z in folds:
        oof[z["val_idx"]] = z["val_approve"]
    y = np.array([int(r["labels"]["approve"]) for r in tr])

    cfg, gb, X = fit_stage(tr, oof)
    stage = ApproveStage(cfg)
    # parity: numpy evaluation == scikit-learn
    ref = gb.decision_function(X[:500])
    mine = np.array([stage.raw(list(x)) for x in X[:500]])
    print("caps", {k: round(v, 1) for k, v in cfg["caps"].items()}, "| parity max |diff| %.2e" % np.abs(ref - mine).max())

    # cross-predicted OOF stage logits for the structural calibration
    caps = estimate_caps(tr)
    Xc = np.array([features(z, r["state"], caps) for z, r in zip(oof, tr)])
    cv = np.zeros(n)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xc, y):
        cv[b] = make_model().fit(Xc[a], y[a]).decision_function(Xc[b])
    pn, ps = 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-cv))
    print("OOF approve  network acc %.4f brier %.4f | staged acc %.4f brier %.4f"
          % (((pn > 0.5) == y).mean(), np.mean((pn - y) ** 2), ((ps > 0.5) == y).mean(), np.mean((ps - y) ** 2)))

    out_f = folds_dir + "_st"
    os.makedirs(out_f, exist_ok=True)
    for f, z in enumerate(folds):
        z = dict(z)
        z["val_approve"] = cv[z["val_idx"]].astype(np.float32)
        z["dev_approve"] = np.array([stage.logit(v, r["state"]) for v, r in zip(z["dev_approve"], dv)], np.float32)
        np.savez(os.path.join(out_f, "fold%d.npz" % f), **z)
    shutil.copy(os.path.join(folds_dir, "args.json"), out_f)

    out_d = full_dir + "_st"
    os.makedirs(out_d, exist_ok=True)
    for name in os.listdir(full_dir):
        if name in ("dev_logits.npz", "policy.json", "struct_policy.json"):
            continue
        dst = os.path.join(out_d, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(full_dir, name)), dst)
    d = dict(np.load(os.path.join(full_dir, "dev_logits.npz")))
    d["approve"] = np.array([stage.logit(v, r["state"]) for v, r in zip(d["approve"], dv)], np.float32)
    np.savez(os.path.join(out_d, "dev_logits.npz"), **d)
    json.dump(cfg, open(os.path.join(out_d, "approve_stage.json"), "w"))
    print("wrote", out_f, out_d)


if __name__ == "__main__":
    main()
