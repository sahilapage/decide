"""
Is adding a decision to the shared model cheap, and does it keep the old answers?

Leave-one-decision-out: a base model is trained on two decisions only, then the
third is added the way a new (fourth) decision would be, with one prompt and one
head, on a label budget. This scores each extended model on dev exactly as
selfcheck.py does (policy cross-fitted on dev halves, like fit_policy_dev.py)
and compares it with the jointly trained reference.

  python tools/extend_eval.py --new grounded --base runs/x_base \
      --ref runs/x_joint runs/x_add_frozen_n300 runs/x_add_tune_n1000 ...

Every run dir holds dev_logits.npz (written by train_mt.py --full) and args.json.
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


def crossfit(dv, R, A, G, repeats=10):
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in dv])
    ya = np.array([int(r["labels"]["approve"]) for r in dv])
    yg = np.array([int(r["labels"]["grounded"]) for r in dv])
    tot = []
    for rep in range(repeats):
        perm = np.random.RandomState(rep).permutation(len(dv))
        h = [perm[: len(dv) // 2], perm[len(dv) // 2:]]
        for k in range(2):
            f, e = h[k], h[1 - k]
            pol = fit_policy(R[f], yr[f], A[f], ya[f], G[f], yg[f], records=[dv[i] for i in f], verbose=False)
            tot.append(score([dv[i] for i in e], pol.responses(R[e], A[e], G[e]))[0])
    acc = {"route": float((R.argmax(1) == yr).mean()), "approve": float(((A > 0) == ya).mean()),
           "grounded": float(((G > 0) == yg).mean())}
    return float(np.mean(tot)), float(np.std(tot)), acc


def describe(run):
    a = json.load(open(os.path.join(run, "args.json")))
    if not a.get("init_from"):
        return "joint (all 3 together)" if a.get("tasks", "route,approve,grounded").count(",") == 2 \
            else "base (%s only)" % a["tasks"]
    mode = "frozen encoder" if a.get("freeze_encoder") else "light tune + replay %.2f" % a.get("replay", 0)
    return "add, %s, %s labels" % (mode, a.get("n_labels") or "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="grounded", choices=["route", "approve", "grounded"])
    ap.add_argument("--base", required=True, help="run trained without the new decision")
    ap.add_argument("--ref", default="", help="jointly trained run, for comparison")
    ap.add_argument("adds", nargs="*")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    dv = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8") if l.strip()]
    base = np.load(os.path.join(args.base, "dev_logits.npz"))
    old = [d for d in ("route", "approve", "grounded") if d != args.new]

    rows = []
    for run in ([args.ref] if args.ref else []) + [args.base] + args.adds:
        z = np.load(os.path.join(run, "dev_logits.npz"))
        L = {d: z[d].astype(np.float64) for d in ("route", "approve", "grounded")}
        drift = ""
        if run not in (args.ref, args.base):
            # old decisions are served from this extended model; how far did they move?
            drift = max(float(np.abs(L[d] - base[d]).max()) for d in old)
        mean, sd, acc = crossfit(dv, L["route"], L["approve"], L["grounded"], args.repeats)
        rows.append((describe(run), mean, sd, acc, drift, os.path.basename(run.rstrip("/"))))

    print("\nnew decision: %s   (old: %s)\n" % (args.new, ", ".join(old)))
    print("%-42s %-14s %8s %8s %8s  %s" % ("model", "dev total (cf)", "route", "approve", "grounded",
                                         "max |old-logit change|"))
    for name, mean, sd, acc, drift, tag in rows:
        print("%-42s %6.2f ± %-5.2f %8.4f %8.4f %8.4f  %s" % (
            name, mean, sd, acc["route"], acc["approve"], acc["grounded"],
            "" if drift == "" else ("0 (bit-identical)" if drift == 0 else "%.4f" % drift)))


if __name__ == "__main__":
    main()
