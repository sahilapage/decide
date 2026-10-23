"""
Second stage of the approve head: a small monotone-free gradient-boosted
correction applied to the shared network's approve logit.

Why: the network sees the amount only through coarse tier x amount buckets
and misses part of the policy near each tier's cap. Out-of-fold, a 60-tree
stage over [network logit, tier, amount / tier cap, amount == 0, the
disqualifying-phrase flags] lifts approve accuracy 88.5% -> 90.4% and the
total score by ~+1.0 (6,500 OOF rows, cross-fitted policy, ~3.6 sd).

It is part of the approve head, not a separate model: its first input is the
shared encoder's approve logit, and route / grounded are untouched. Caps are
estimated from the training labels (the amount where approval falls to 50%).

Serving needs numpy only: the trees are stored as flat arrays in JSON and
evaluated here (~0.1 ms). Fitting (fit_stage) uses scikit-learn.
"""

import json
import re
from typing import Any, Dict, List

import numpy as np

TIERS = ["free", "pro", "enterprise"]
PHRASES = [r"raised something similar", r"signed up recently", r"outside your usual policy|appreciate an exception",
           r"with my bank|dispute a charge", r"question is"]
_PH = [re.compile(p) for p in PHRASES]
FEATURES = ["net_logit", "tier", "amount_over_cap", "amount_is_zero"] + ["phrase_%d" % i for i in range(len(PHRASES))]


def _amount(state):
    try:
        return max(0.0, float(state.get("amount") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def features(z: float, state: Dict[str, Any], caps: Dict[str, float]) -> List[float]:
    tier = str(state.get("customer_tier") or "free").lower()
    ti = TIERS.index(tier) if tier in TIERS else 0
    a = _amount(state)
    text = (str(state.get("subject") or "") + " " + str(state.get("body") or "")).lower()
    return [float(z), float(ti), a / caps[TIERS[ti]] if a > 0 else 0.0, float(a == 0)] + \
        [1.0 if p.search(text) else 0.0 for p in _PH]


class ApproveStage:
    def __init__(self, cfg: Dict[str, Any]):
        self.caps = cfg["caps"]
        self.base = float(cfg["baseline"])
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in cfg["trees"]]

    @classmethod
    def load(cls, path: str) -> "ApproveStage":
        with open(path) as fh:
            return cls(json.load(fh))

    def raw(self, x: List[float]) -> float:
        s = self.base
        for t in self.trees:
            i = 0
            while not t["leaf"][i]:
                v = x[t["feat"][i]]
                i = t["left"][i] if v <= t["thr"][i] else t["right"][i]
            s += t["value"][i]
        return s

    def logit(self, z: float, state: Dict[str, Any]) -> float:
        return self.raw(features(z, state, self.caps))


def estimate_caps(rows) -> Dict[str, float]:
    """Per tier, the amount where approval (no disqualifying phrase, amount > 0) falls to 50%."""
    from sklearn.linear_model import LogisticRegression
    caps = {}
    for tier in TIERS:
        xs, ys = [], []
        for r in rows:
            s = r["state"]
            text = (s["subject"] + " " + s["body"]).lower()
            if s["customer_tier"] == tier and _amount(s) > 0 and not any(p.search(text) for p in _PH):
                xs.append(np.log(_amount(s)))
                ys.append(int(r["labels"]["approve"]))
        m = LogisticRegression(C=100.0, max_iter=5000).fit(np.array(xs)[:, None], np.array(ys))
        caps[tier] = float(np.exp(-m.intercept_[0] / m.coef_[0, 0]))
    return caps


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def export(gb, caps) -> Dict[str, Any]:
    trees = []
    for (pred,) in gb._predictors:
        nd = pred.nodes
        trees.append(dict(feat=nd["feature_idx"].astype(int).tolist(), thr=nd["num_threshold"].astype(float).tolist(),
                          left=nd["left"].astype(int).tolist(), right=nd["right"].astype(int).tolist(),
                          leaf=nd["is_leaf"].astype(bool).tolist(), value=nd["value"].astype(float).tolist()))
    return dict(caps=caps, baseline=float(np.ravel(gb._baseline_prediction)[0]), trees=trees, features=FEATURES)


def fit_stage(rows, net_logits):
    """Fit on OUT-OF-FOLD network logits (so the stage learns the network's real errors)."""
    caps = estimate_caps(rows)
    X = np.array([features(z, r["state"], caps) for z, r in zip(net_logits, rows)])
    y = np.array([int(r["labels"]["approve"]) for r in rows])
    gb = make_model().fit(X, y)
    return export(gb, caps), gb, X
