"""
How much should we abstain? Sweep the coverage floor used when tuning the
abstain thresholds and measure the cross-fitted dev score (policy fitted on
one half, scored on the other, same halves for every floor).

The trade-off: answering fewer cases raises answered-set quality, but if the
held-out coverage falls below the 80% target, quality is scaled down by
coverage/0.8. So we also report the lowest held-out coverage seen.

  python tools/coverage_sweep.py runs/v6t2h_e17_s3/dev_logits.npz --struct runs/v6t2h_e17_s3/struct_policy.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev_logits")
    ap.add_argument("--struct", default="")
    ap.add_argument("--floors", default="0.80,0.81,0.82,0.83,0.84,0.85,0.87,0.90")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    struct = json.load(open(args.struct)) if args.struct else None
    dv = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8") if l.strip()]
    z = np.load(args.dev_logits)
    R, A, G = (z[k].astype(np.float64) for k in ("route", "approve", "grounded"))
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in dv])
    ya = np.array([int(r["labels"]["approve"]) for r in dv])
    yg = np.array([int(r["labels"]["grounded"]) for r in dv])
    base = None
    for floor in [float(x) for x in args.floors.split(",")]:
        tots, mincov, parts = [], 1.0, []
        for rep in range(args.repeats):
            perm = np.random.RandomState(rep).permutation(len(dv))
            halves = [perm[: len(dv) // 2], perm[len(dv) // 2:]]
            for k in range(2):
                f, e = halves[k], halves[1 - k]
                pol = fit_policy(R[f], yr[f], A[f], ya[f], G[f], yg[f], records=[dv[i] for i in f],
                                 coverage=floor, verbose=False, calib="noise", struct_cfg=struct)
                t, p, d = score([dv[i] for i in e], pol.responses(R[e], A[e], G[e]))
                tots.append(t)
                parts.append(p)
                mincov = min(mincov, d["route_cov"], d["appr_cov"], d["gr_cov"])
        tots = np.array(tots)
        if base is None:
            base = tots
        q = np.mean([p["quality"] for p in parts])
        print("floor %.2f  cross-fit %.2f (sd %.2f)  vs first %+.2f  quality %.4f  min held-out coverage %.3f"
              % (floor, tots.mean(), tots.std(), (tots - base).mean(), q, mincov), flush=True)


if __name__ == "__main__":
    main()
