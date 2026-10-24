"""
Combine fold outputs of one or more runs, fit the policy, score.

  python training/eval_runs.py runs/e32                 # one run
  python training/eval_runs.py runs/a runs/b --save_soft runs/teacher_soft.npz   # ensemble -> teacher targets

Logits of several runs are averaged (route in log-prob space, binaries in
logit space). Dev logits are the mean over the fold models.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import glob
import json
import os

import numpy as np

from decide.policy import fit_policy, sigmoid, softmax
from decide.scoring import score

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def collect(run, n_train):
    files = sorted(glob.glob(os.path.join(run, "fold*.npz")))
    oof = {"route": np.zeros((n_train, 5)), "approve": np.zeros(n_train), "grounded": np.zeros(n_train)}
    seen = np.zeros(n_train, bool)
    dev = None
    for f in files:
        z = np.load(f)
        i = z["val_idx"]
        seen[i] = True
        rl = z["val_route"]
        oof["route"][i] = rl - np.log(np.exp(rl - rl.max(1, keepdims=True)).sum(1, keepdims=True)) - rl.max(1, keepdims=True)
        oof["approve"][i] = z["val_approve"]
        oof["grounded"][i] = z["val_grounded"]
        d = {"route": z["dev_route"], "approve": z["dev_approve"], "grounded": z["dev_grounded"]}
        d["route"] = d["route"] - np.log(np.exp(d["route"] - d["route"].max(1, keepdims=True)).sum(1, keepdims=True)) - d["route"].max(1, keepdims=True)
        dev = {k: v / len(files) for k, v in d.items()} if dev is None else {k: dev[k] + d[k] / len(files) for k in d}
    return oof, dev, seen, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--save_soft", default="")
    ap.add_argument("--save_policy", default="")
    args = ap.parse_args()
    tr = load("data/train.jsonl")
    dv = load("data/dev.jsonl")
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in tr])
    ya = np.array([int(r["labels"]["approve"]) for r in tr])
    yg = np.array([int(r["labels"]["grounded"]) for r in tr])

    oofs, devs, seen_all = [], [], None
    for run in args.runs:
        o, d, seen, nf = collect(run, len(tr))
        print("%s: %d folds" % (run, nf))
        oofs.append(o)
        devs.append(d)
        seen_all = seen if seen_all is None else seen_all & seen
    avg = lambda xs, k: sum(x[k] for x in xs) / len(xs)
    oof = {k: avg(oofs, k) for k in oofs[0]}
    dev = {k: avg(devs, k) for k in devs[0]}
    m = seen_all
    if not m.all():
        print("  note: only %d / %d train rows have OOF predictions" % (m.sum(), len(tr)))
    trm = [r for r, s in zip(tr, m) if s]

    pol = fit_policy(oof["route"][m], yr[m], oof["approve"][m], ya[m], oof["grounded"][m], yg[m], records=trm)
    print("OOF:")
    score(trm, pol.responses(oof["route"][m], oof["approve"][m], oof["grounded"][m]), verbose=True)
    print("DEV (mean of fold models):")
    total, parts, det = score(dv, pol.responses(dev["route"], dev["approve"], dev["grounded"]), verbose=True)
    print("policy:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in pol.cfg.items()}))

    if args.save_policy:
        pol.save(args.save_policy)
    if args.save_soft:
        np.savez(args.save_soft,
                 route_prob=pol.route_probs(oof["route"]),
                 approve_prob=pol.p_approve(oof["approve"]),
                 grounded_prob=pol.p_grounded(oof["grounded"]),
                 seen=m)
        print("saved teacher targets to", args.save_soft)


if __name__ == "__main__":
    main()
