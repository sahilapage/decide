"""
Does averaging several seeds' logits beat one seed? Cross-fitted on dev, with
the SAME random halves for every variant so the comparison is paired.

  python tools/ensemble_eval.py --struct runs/v5h_e17_s3/struct_policy.json \
      runs/v5h_e17_s0 runs/v5h_e17_s1 runs/v5h_e17_s2 runs/v5h_e17_s3
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def crossfit(dv, R, A, G, yr, ya, yg, struct, repeats):
    totals, parts_all = [], []
    for rep in range(repeats):
        perm = np.random.RandomState(rep).permutation(len(dv))
        halves = [perm[: len(dv) // 2], perm[len(dv) // 2:]]
        for k in range(2):
            f, e = halves[k], halves[1 - k]
            pol = fit_policy(R[f], yr[f], A[f], ya[f], G[f], yg[f], records=[dv[i] for i in f],
                             verbose=False, calib="noise", struct_cfg=struct)
            t, parts, _ = score([dv[i] for i in e], pol.responses(R[e], A[e], G[e]))
            totals.append(t)
            parts_all.append(parts)
    mp = {k: float(np.mean([p[k] for p in parts_all])) for k in parts_all[0]}
    return np.array(totals), mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--struct", default="")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    struct = json.load(open(args.struct)) if args.struct else None
    dv = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8") if l.strip()]
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in dv])
    ya = np.array([int(r["labels"]["approve"]) for r in dv])
    yg = np.array([int(r["labels"]["grounded"]) for r in dv])
    L = [np.load(os.path.join(r, "dev_logits.npz")) for r in args.runs]
    names = [os.path.basename(r.rstrip("/")) for r in args.runs]

    results = {}
    for size in range(1, len(L) + 1):
        for combo in itertools.combinations(range(len(L)), size):
            R = np.mean([L[i]["route"].astype(np.float64) for i in combo], 0)
            A = np.mean([L[i]["approve"].astype(np.float64) for i in combo], 0)
            G = np.mean([L[i]["grounded"].astype(np.float64) for i in combo], 0)
            tot, mp = crossfit(dv, R, A, G, yr, ya, yg, struct, args.repeats)
            key = "+".join(names[i].split("_")[-1] for i in combo)
            results[key] = tot
            print("%-14s n=%d  cross-fit %.2f (sd %.2f) | quality %.4f calib %.4f abst %.4f"
                  % (key, size, tot.mean(), tot.std(), mp["quality"], mp["calibration"], mp["abstention"]),
                  flush=True)

    singles = [k for k in results if "+" not in k]
    base = np.mean([results[k] for k in singles], 0)
    print("\nmean single seed: %.2f" % base.mean())
    for k, v in results.items():
        if "+" in k:
            d = v - base
            print("%-14s vs mean single: %+.2f (paired sd %.2f)" % (k, d.mean(), d.std()))


if __name__ == "__main__":
    main()
