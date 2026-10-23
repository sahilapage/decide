"""
Compare calibration recipes on out-of-fold predictions (6,500 rows), with
cross-fitting: fit the policy on one half, score the other half, swap,
repeated over random splits. Much less noisy than dev's 1,500 rows.

  python tools/crossfit_oof.py runs/e32s [more runs to average] --calib basic,noise
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402
from eval_runs import QUEUES, collect, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--calib", default="basic,noise")
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    tr = load("data/train.jsonl")
    oofs = [collect(r, len(tr))[0] for r in args.runs]
    oof = {k: sum(o[k] for o in oofs) / len(oofs) for k in oofs[0]}
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])
    for calib in args.calib.split(","):
        tots, parts_all = [], []
        for rep in range(args.repeats):
            perm = np.random.RandomState(rep).permutation(len(tr))
            halves = [perm[: len(tr) // 2], perm[len(tr) // 2:]]
            # each held-out half is scored on its own, with the one policy fitted
            # on the other half - exactly how the deployed service is scored
            for k in range(2):
                f, e = halves[k], halves[1 - k]
                pol = fit_policy(oof["route"][f], yr[f], oof["approve"][f], ya[f], oof["grounded"][f], yg[f],
                                 records=[tr[i] for i in f], verbose=False, calib=calib)
                o = pol.responses(oof["route"][e], oof["approve"][e], oof["grounded"][e])
                t, parts, det = score([tr[i] for i in e], o)
                tots.append(t)
                parts_all.append(parts)
        mp = {k: np.mean([p[k] for p in parts_all]) for k in parts_all[0]}
        print("%-6s total %.2f (+/- %.2f) | quality %.4f calib %.4f abstention %.4f"
              % (calib, np.mean(tots), np.std(tots), mp["quality"], mp["calibration"], mp["abstention"]), flush=True)


if __name__ == "__main__":
    main()
