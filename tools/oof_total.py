"""
Low-noise recipe comparison: the selfcheck total on the 6,500 out-of-fold
train rows (policy cross-fitted on halves, 10 repeats, the same halves for
every run), with and without the approve second stage (itself 5-fold
cross-predicted, so no row is scored by a stage that saw it).

  python tools/oof_total.py runs/v6t2_e17 runs/xE1_e17 runs/xE2_e17 ...
The first run is the reference for the paired differences.
"""

import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.approve_stage import estimate_caps, features, make_model  # noqa: E402
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load_oof(run, n):
    R, A, G = np.zeros((n, 5)), np.zeros(n), np.zeros(n)
    for f in range(5):
        z = np.load(os.path.join(run, "fold%d.npz" % f))
        i = z["val_idx"]
        R[i], A[i], G[i] = z["val_route"], z["val_approve"], z["val_grounded"]
    return R, A, G


def main():
    runs = [r for r in sys.argv[1:] if os.path.exists(os.path.join(r, "fold4.npz"))]
    tr = [json.loads(l) for l in open("data/train.jsonl")]
    n = len(tr)
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])
    caps = estimate_caps(tr)
    halves = []
    for rep in range(10):
        perm = np.random.RandomState(rep).permutation(n)
        halves += [(perm[: n // 2], perm[n // 2:]), (perm[n // 2:], perm[: n // 2])]

    def total(R, A, G):
        out = []
        for f, e in halves:
            pol = fit_policy(R[f], yr[f], A[f], ya[f], G[f], yg[f], records=[tr[i] for i in f],
                             verbose=False, calib="noise")
            out.append(score([tr[i] for i in e], pol.responses(R[e], A[e], G[e])))
        return np.array([t for t, _, _ in out]), {k: np.mean([p[k] for _, p, _ in out]) for k in out[0][1]}

    ref = {}
    for run in runs:
        R, A, G = load_oof(run, n)
        X = np.array([features(z, r["state"], caps) for z, r in zip(A, tr)])
        S = np.zeros(n)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(X, ya):
            S[b] = make_model().fit(X[a], ya[a]).decision_function(X[b])
        for tag, AA in (("raw", A), ("stage", S)):
            t, p = total(R, AA, G)
            if tag not in ref:
                ref[tag] = t
            d = t - ref[tag]
            print("%-16s %-5s OOF total %.2f  vs ref %+.2f (paired sd %.2f) | q %.4f cal %.4f abst %.4f | "
                  "route acc %.4f grounded acc %.4f"
                  % (os.path.basename(run), tag, t.mean(), d.mean(), d.std(), p["quality"], p["calibration"],
                     p["abstention"], (R.argmax(1) == yr).mean(), ((G > 0) == yg).mean()), flush=True)


if __name__ == "__main__":
    main()
