"""
Train the gate-1 baseline: shared sparse backbone + three heads.

  python training/train_baseline.py

5-fold out-of-fold logits on train fit the calibration and abstain policy;
the final model is refit on all of train. Dev is only used to report.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import json
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold

from decide.baseline import QUEUES, BaselineModel
from decide.policy import fit_policy
from decide.scoring import score


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    tr = load("data/train.jsonl")
    dv = load("data/dev.jsonl")
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])

    oof_r = np.zeros((len(tr), 5))
    oof_a = np.zeros(len(tr))
    oof_g = np.zeros(len(tr))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for f, (a, b) in enumerate(skf.split(tr, yr)):
        m = BaselineModel().fit([tr[i] for i in a])
        r, ap, g = m.logits([tr[i]["state"] for i in b])
        oof_r[b], oof_a[b], oof_g[b] = r, ap, g
        print("fold", f, "done")

    pol = fit_policy(oof_r, yr, oof_a, ya, oof_g, yg, records=tr)
    print("OOF (train):")
    score(tr, pol.responses(oof_r, oof_a, oof_g), verbose=True)

    m = BaselineModel().fit(tr)
    r, a, g = m.logits([x["state"] for x in dv])
    print("DEV:")
    score(dv, pol.responses(r, a, g), verbose=True)

    with open("artifacts/baseline.pkl", "wb") as fh:
        pickle.dump(m, fh)
    pol.save("artifacts/baseline_policy.json")
    print("saved artifacts/baseline.pkl, artifacts/baseline_policy.json")


if __name__ == "__main__":
    main()
