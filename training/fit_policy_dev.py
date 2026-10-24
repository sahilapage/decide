"""
Fit the serving policy on the DEPLOYED model's own dev predictions.

The full model is trained on all of train and has never seen dev, so dev is a
clean held-out calibration set for exactly the network we ship. To report an
honest dev number we cross-fit: fit the policy on one half of dev, score the
other half, and swap (repeated over several random splits). The final policy
is then fitted on all of dev.

  python training/fit_policy_dev.py runs/e32_full/dev_logits.npz --out runs/e32_full/policy.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json

import numpy as np

from decide.policy import fit_policy
from decide.scoring import score

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev_logits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calib", default="noise", choices=["noise", "basic"])
    ap.add_argument("--struct", default="", help="policy json fitted on OOF: keep its noise params, refit scales on dev")
    args = ap.parse_args()
    struct = json.load(open(args.struct)) if args.struct else None
    dv = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8") if l.strip()]
    z = np.load(args.dev_logits)
    R, A, G = z["route"].astype(np.float64), z["approve"].astype(np.float64), z["grounded"].astype(np.float64)
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in dv])
    ya = np.array([int(r["labels"]["approve"]) for r in dv])
    yg = np.array([int(r["labels"]["grounded"]) for r in dv])

    # cross-fitted estimate: assemble out-of-half responses, score the whole of dev
    totals = []
    for rep in range(args.repeats):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(len(dv))
        halves = [perm[: len(dv) // 2], perm[len(dv) // 2:]]
        # each held-out half scored on its own with the one policy fitted on the other half
        for k in range(2):
            fit_i, ev_i = halves[k], halves[1 - k]
            pol = fit_policy(R[fit_i], yr[fit_i], A[fit_i], ya[fit_i], G[fit_i], yg[fit_i],
                             records=[dv[i] for i in fit_i], verbose=False, calib=args.calib,
                             struct_cfg=struct)
            o = pol.responses(R[ev_i], A[ev_i], G[ev_i])
            t, parts, det = score([dv[i] for i in ev_i], o, verbose=(rep == 0 and k == 0))
            totals.append(t)
    print("cross-fitted dev total: mean %.2f  (min %.2f, max %.2f over %d halves)"
          % (np.mean(totals), np.min(totals), np.max(totals), len(totals)))

    pol = fit_policy(R, yr, A, ya, G, yg, records=dv, verbose=True, calib=args.calib, struct_cfg=struct)
    print("in-sample dev (optimistic):")
    score(dv, pol.responses(R, A, G), verbose=True)
    pol.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
