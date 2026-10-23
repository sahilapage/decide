"""
Approve: is there headroom left above the network's own head?

(B) Stacking: a small gradient-boosted model over [network approve logit, tier,
    amount, amount / tier cap, disqualifying-phrase flags], trained on the
    5-fold OUT-OF-FOLD network logits (so it learns the network's real
    mistakes), applied to the shipped network's dev logits. Scored with the
    usual cross-fitted policy, same halves as the baseline.
(C) Irreducibility: among under-cap tickets with no disqualifying phrase,
    can bag-of-n-grams predict the refusals at all? (5-fold AUC on train.)

  python tools/approve_stack.py runs/v6t2_e17 runs/v6t2h_e17_s3
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.policy import fit_policy  # noqa: E402
from decide.scoring import score  # noqa: E402

QUEUES = ["billing", "technical", "account", "shipping", "general"]
TIERS = ["free", "pro", "enterprise"]
CAPS = {"free": 25.0, "pro": 110.0, "enterprise": 385.0}
PHRASES = [r"raised something similar", r"signed up recently", r"outside your usual policy|appreciate an exception",
           r"with my bank|dispute a charge", r"question is"]


def feats(rows, z):
    X = []
    for r, zz in zip(rows, z):
        s = r["state"]
        t = (s["subject"] + " " + s["body"]).lower()
        a = float(s["amount"] or 0)
        tier = s["customer_tier"]
        X.append([zz, TIERS.index(tier), a, np.log1p(a), a / CAPS[tier], float(a == 0)]
                 + [float(bool(re.search(p, t))) for p in PHRASES])
    return np.array(X)


def crossfit(dv, R, A, G, ys, struct, reps=10):
    yr, ya, yg = ys
    tots = []
    for rep in range(reps):
        perm = np.random.RandomState(rep).permutation(len(dv))
        halves = [perm[: len(dv) // 2], perm[len(dv) // 2:]]
        for k in range(2):
            f, e = halves[k], halves[1 - k]
            pol = fit_policy(R[f], yr[f], A[f], ya[f], G[f], yg[f], records=[dv[i] for i in f],
                             verbose=False, calib="noise", struct_cfg=struct)
            tots.append(score([dv[i] for i in e], pol.responses(R[e], A[e], G[e]))[0])
    return np.array(tots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof_run")
    ap.add_argument("full_run")
    args = ap.parse_args()
    tr = [json.loads(l) for l in open("data/train.jsonl")]
    dv = [json.loads(l) for l in open("data/dev.jsonl")]
    n = len(tr)
    oof = np.zeros(n)
    for f in range(5):
        z = np.load(os.path.join(args.oof_run, "fold%d.npz" % f))
        oof[z["val_idx"]] = z["val_approve"]
    ya_tr = np.array([int(r["labels"]["approve"]) for r in tr])
    zd = np.load(os.path.join(args.full_run, "dev_logits.npz"))
    R, A, G = (zd[k].astype(np.float64) for k in ("route", "approve", "grounded"))
    ys = (np.array([QUEUES.index(r["labels"]["route"]) for r in dv]),
          np.array([int(r["labels"]["approve"]) for r in dv]),
          np.array([int(r["labels"]["grounded"]) for r in dv]))
    struct = json.load(open(os.path.join(args.full_run, "struct_policy.json")))

    # (B) stacking
    Xtr, Xdv = feats(tr, oof), feats(dv, A)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                        min_samples_leaf=40, l2_regularization=1.0, categorical_features=[1],
                                        random_state=0)
    # OOF check of the stacker itself on train (does it beat the network's logit?)
    p_oof = cross_val_predict(gb, Xtr, ya_tr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    p_net = 1 / (1 + np.exp(-oof))
    br = lambda p, y: float(np.mean((p - y) ** 2))
    print("(B) train OOF  network: acc %.4f brier %.4f | stacked: acc %.4f brier %.4f"
          % (((p_net > 0.5) == ya_tr).mean(), br(p_net, ya_tr), ((p_oof > 0.5) == ya_tr).mean(), br(p_oof, ya_tr)))
    gb.fit(Xtr, ya_tr)
    p_dev = np.clip(gb.predict_proba(Xdv)[:, 1], 1e-6, 1 - 1e-6)
    A_stack = np.log(p_dev / (1 - p_dev))
    base = crossfit(dv, R, A, G, ys, struct)
    stack = crossfit(dv, R, A_stack, G, ys, struct)
    d = stack - base
    print("(B) dev cross-fit  baseline %.2f  stacked %.2f  diff %+.2f (paired sd %.2f)"
          % (base.mean(), stack.mean(), d.mean(), d.std()))

    # (C) irreducibility of the hidden refusals
    idx = [i for i, r in enumerate(tr) if 0 < r["state"]["amount"] < 0.8 * CAPS[r["state"]["customer_tier"]]
           and not any(re.search(p, (r["state"]["subject"] + " " + r["state"]["body"]).lower()) for p in PHRASES)]
    texts = [tr[i]["state"]["subject"] + " " + tr[i]["state"]["body"] for i in idx]
    y = ya_tr[idx]
    Xt = CountVectorizer(ngram_range=(1, 3), min_df=3, binary=True).fit_transform(texts)
    p = cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), Xt, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print("(C) under-cap, no phrase: n=%d refusal rate %.3f | n-gram AUC %.3f | network OOF AUC %.3f"
          % (len(idx), 1 - y.mean(), roc_auc_score(y, p), roc_auc_score(y, oof[idx])))


if __name__ == "__main__":
    main()
