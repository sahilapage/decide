"""
How close is the model to the best score this data allows?

Uses 5-fold out-of-fold predictions on train (6,500 rows, never seen by the
fold's model) and asks, per decision:

  * error among the k% most confident cases. If labels carry noise that the
    text cannot reveal, this curve goes FLAT: the most confident 10% are wrong
    as often as the most confident 80%, and no confidence signal can rank
    those errors.
  * the best abstention score any model could get, 1 - floor / full_error
    (the selfcheck abstention metric with the floor as risk@80%).

For approve, the floor comes from a "perfect except hidden history" model:
every case the policy would approve is refused with probability h (estimated
from the data as 1 - the calibrated ceiling), every other case is certain.

  python tools/ceilings.py runs/v5_e17 [--policy runs/v5h_e17_s3/struct_policy.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def curve(conf, err, qs=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    o = np.argsort(-conf)
    return {q: float(err[o[: int(q * len(err))]].mean()) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof_run")
    ap.add_argument("--policy", default="")
    args = ap.parse_args()
    tr = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8") if l.strip()]
    n = len(tr)
    R, A, G = np.zeros((n, 5)), np.zeros(n), np.zeros(n)
    for f in range(5):
        z = np.load(os.path.join(args.oof_run, "fold%d.npz" % f))
        i = z["val_idx"]
        R[i], A[i], G[i] = z["val_route"], z["val_approve"], z["val_grounded"]
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])

    pr = np.exp(R - R.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    out = {}

    # route: flat error plateau = label flips the text cannot reveal
    er = (pr.argmax(1) != yr).astype(float)
    cr = curve(pr.max(1), er)
    floor_r = cr[0.2]                       # plateau level (top-20% most confident)
    out["route"] = dict(curve=cr, full_error=float(er.mean()), floor=floor_r,
                        abst_ours=1 - cr[0.8] / er.mean(), abst_ceiling=1 - floor_r / er.mean())

    # approve: hidden-history refusals
    pa = 1 / (1 + np.exp(-A))
    ea = ((pa > 0.5) != ya).astype(float)
    ca = curve(np.maximum(pa, 1 - pa), ea)
    h = 0.144
    if args.policy:
        h = 1 - json.load(open(args.policy))["hi_appr"]
    b = ya.mean()                           # approve base rate
    approvable = min(1.0, b / (1 - h))      # share the visible policy would approve
    certain = 1 - approvable
    risk80 = max(0.0, 0.8 - certain) * h / 0.8
    full_oracle = approvable * h
    out["approve"] = dict(curve=ca, full_error=float(ea.mean()), hidden_refusal_rate=h, base_rate=float(b),
                          oracle_full_error=full_oracle, oracle_risk80=risk80,
                          abst_ours=1 - ca[0.8] / ea.mean(), abst_ceiling_oracle=1 - risk80 / full_oracle)

    # grounded
    pg = 1 / (1 + np.exp(-G))
    eg = ((pg > 0.5) != yg).astype(float)
    out["grounded"] = dict(curve=curve(np.maximum(pg, 1 - pg), eg), full_error=float(eg.mean()))

    for d, v in out.items():
        print("%s: full error %.4f" % (d, v["full_error"]))
        print("   error among most confident: " + "  ".join("%d%% %.4f" % (q * 100, e) for q, e in v["curve"].items()))
        for k in v:
            if k not in ("curve", "full_error"):
                print("   %-22s %.4f" % (k, v[k]))
    json.dump(out, open("docs/ceilings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
