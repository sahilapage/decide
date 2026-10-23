"""
Low-noise comparison of approve second stages: total selfcheck score on the
6,500 out-of-fold train rows (policy cross-fitted on halves, 10 repeats),
with each second stage itself cross-validated (5-fold) so no row is scored by
a stage that saw it.

  python tools/approve_variants_oof.py runs/v6t2_e17
"""

import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402

QUEUES = ["billing", "technical", "account", "shipping", "general"]
T = ["free", "pro", "enterprise"]
CAPS = {"free": 25.0, "pro": 110.0, "enterprise": 385.0}
PHRASES = [r"raised something similar", r"signed up recently", r"outside your usual policy|appreciate an exception",
           r"with my bank|dispute a charge", r"question is"]


def main():
    run = sys.argv[1]
    tr = [json.loads(l) for l in open("data/train.jsonl")]
    n = len(tr)
    R, A, G = np.zeros((n, 5)), np.zeros(n), np.zeros(n)
    for f in range(5):
        z = np.load(os.path.join(run, "fold%d.npz" % f))
        i = z["val_idx"]
        R[i], A[i], G[i] = z["val_route"], z["val_approve"], z["val_grounded"]
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])
    tier = np.array([T.index(r["state"]["customer_tier"]) for r in tr])
    amt = np.array([float(r["state"]["amount"] or 0) for r in tr])
    ratio = np.where(amt > 0, amt / np.array([CAPS[T[t]] for t in tier]), 0.0)
    text = [(r["state"]["subject"] + " " + r["state"]["body"]).lower() for r in tr]
    ph = np.array([[float(bool(re.search(p, t))) for p in PHRASES] for t in text])
    a0 = (amt == 0).astype(float)

    def cvpred(make, X):
        out = np.zeros(n)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(X, ya):
            m = make().fit(X[a], ya[a])
            p = np.clip(m.predict_proba(X[b])[:, 1], 1e-6, 1 - 1e-6)
            out[b] = np.log(p / (1 - p))
        return out

    gbm = lambda cat, mono=None, it=60: (lambda: HistGradientBoostingClassifier(
        max_iter=it, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0,
        random_state=0, categorical_features=cat, monotonic_cst=mono))
    variants = {
        "network only": A,
        "GBM logit+tier+ratio (mono)": cvpred(gbm([1], [1, 0, -1]), np.column_stack([A, tier, ratio])),
        "GBM logit+tier+ratio+phrases": cvpred(gbm([1]), np.column_stack([A, tier, ratio, a0, ph])),
        "GBM all 11 (200 trees)": cvpred(gbm([1], None, 200), np.column_stack([A, tier, amt, np.log1p(amt), ratio, a0, ph])),
        "LR logit+phrases+ratio hinges": cvpred(lambda: LogisticRegression(C=1.0, max_iter=5000), np.column_stack(
            [A, a0, ph] + [np.maximum(0, ratio - k) for k in (0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)] + [np.minimum(ratio, 2)])),
    }
    base = None
    for name, AA in variants.items():
        tots = []
        for rep in range(10):
            perm = np.random.RandomState(rep).permutation(n)
            halves = [perm[: n // 2], perm[n // 2:]]
            for k in range(2):
                f, e = halves[k], halves[1 - k]
                pol = fit_policy(R[f], yr[f], AA[f], ya[f], G[f], yg[f], records=[tr[i] for i in f],
                                 verbose=False, calib="noise")
                tots.append(score([tr[i] for i in e], pol.responses(R[e], AA[e], G[e]))[0])
        tots = np.array(tots)
        if base is None:
            base = tots
        p = 1 / (1 + np.exp(-AA))
        print("%-32s OOF total %.2f  diff %+.2f (paired sd %.2f) | acc %.4f brier %.4f"
              % (name, tots.mean(), (tots - base).mean(), (tots - base).std(), ((p > 0.5) == ya).mean(),
                 np.mean((p - ya) ** 2)), flush=True)


if __name__ == "__main__":
    main()
