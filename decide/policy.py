"""
Calibration + commit + abstain policy, shared by every backbone.

The backbone produces raw logits per decision. This module turns them into
calibrated probabilities (one temperature per decision, fitted on out-of-fold
predictions), picks the committed value, and decides when to abstain.

  route     probs = softmax(logits / T_route); value = argmax;
            abstain when max prob < tau_route
  approve   p = sigmoid(logit / T_appr + b); value = p > 0.75 (3:1 cost
            break-even: approving costs 3(1-p), holding costs p);
            abstain when expected cost min(3(1-p), p) > tau_appr
  grounded  p = sigmoid(logit / T_gr + b); value = p > 0.5;
            abstain when max(p, 1-p) < tau_gr

Thresholds are tuned against the offline copy of the judges' scorer, with
coverage kept just above the 80% target.
"""

import json
import math
from typing import Dict, List

import numpy as np

QUEUES = ["billing", "technical", "account", "shipping", "general"]
TARGET_COVERAGE = 0.815


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- fitting
def fit_temperature_multiclass(logits: np.ndarray, y: np.ndarray) -> float:
    best, bt = 1e18, 1.0
    for t in np.exp(np.linspace(math.log(0.05), math.log(20), 400)):
        p = softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best:
            best, bt = nll, t
    return float(bt)


def fit_platt(logit: np.ndarray, y: np.ndarray):
    """Fit p = sigmoid(a * logit + b) by Newton steps on NLL."""
    from scipy.optimize import minimize

    x = np.asarray(logit, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    def nll(w):
        z = w[0] * x + w[1]
        # log(1 + e^z) - y z, computed stably
        return float((np.logaddexp(0.0, z) - yy * z).mean())

    def grad(w):
        p = sigmoid(w[0] * x + w[1])
        return np.array([((p - yy) * x).mean(), (p - yy).mean()])

    r = minimize(nll, np.array([1.0, 0.0]), jac=grad, method="L-BFGS-B",
                 bounds=[(1e-3, 20.0), (-20.0, 20.0)])
    return float(r.x[0]), float(r.x[1])


class Policy:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    # ---- apply ------------------------------------------------------------
    def route(self, logits: np.ndarray):
        probs = softmax(np.asarray(logits, dtype=np.float64) / self.cfg["T_route"])
        k = int(probs.argmax())
        conf = float(probs[k])
        pd = {q: float(p) for q, p in zip(QUEUES, probs)}
        return QUEUES[k], pd, conf, conf < self.cfg["tau_route"]

    def approve(self, logit: float):
        p = float(sigmoid(self.cfg["a_appr"] * logit + self.cfg["b_appr"]))
        value = p > self.cfg["thr_appr"]
        if self.cfg.get("appr_mode", "cost") == "conf":
            return value, p, max(p, 1.0 - p) < self.cfg["tau_appr"]
        risk = min(3.0 * (1.0 - p), p)
        return value, p, risk > self.cfg["tau_appr"]

    def grounded(self, logit: float):
        p = float(sigmoid(self.cfg["a_gr"] * logit + self.cfg["b_gr"]))
        value = p > self.cfg["thr_gr"]
        return value, p, max(p, 1.0 - p) < self.cfg["tau_gr"]

    # ---- batch helpers (offline scoring) -----------------------------------
    def responses(self, route_logits, appr_logits, gr_logits):
        outs = {"route": [], "approve": [], "grounded": []}
        for z in route_logits:
            v, pd, c, a = self.route(z)
            outs["route"].append({"value": v, "probs": pd, "confidence": c, "abstain": a})
        for z in appr_logits:
            v, p, a = self.approve(float(z))
            outs["approve"].append({"value": v, "p_true": p, "abstain": a})
        for z in gr_logits:
            v, p, a = self.grounded(float(z))
            outs["grounded"].append({"value": v, "p_true": p, "abstain": a})
        return outs

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.cfg, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))


def fit_policy(route_logits, y_route, appr_logits, y_appr, gr_logits, y_gr,
               records=None, coverage=TARGET_COVERAGE, verbose=True) -> Policy:
    """Fit calibration on OOF logits, then pick abstain thresholds for coverage."""
    route_logits = np.asarray(route_logits, dtype=np.float64)
    y_route = np.asarray(y_route)
    T = fit_temperature_multiclass(route_logits, y_route)
    a_ap, b_ap = fit_platt(np.asarray(appr_logits), np.asarray(y_appr))
    a_gr, b_gr = fit_platt(np.asarray(gr_logits), np.asarray(y_gr))

    pr = softmax(route_logits / T).max(1)
    pa = sigmoid(a_ap * np.asarray(appr_logits) + b_ap)
    pg = sigmoid(a_gr * np.asarray(gr_logits) + b_gr)
    q = 1.0 - coverage
    cfg = {
        "T_route": T, "a_appr": a_ap, "b_appr": b_ap, "a_gr": a_gr, "b_gr": b_gr,
        "thr_appr": 0.75, "thr_gr": 0.5,
        # abstain on the lowest-confidence (1-coverage) share
        "tau_route": float(np.quantile(pr, q)),
        "tau_appr": float(np.quantile(np.minimum(3 * (1 - pa), pa), 1 - q)),
        "tau_gr": float(np.quantile(np.maximum(pg, 1 - pg), q)),
    }
    pol = Policy(cfg)
    if records is not None:
        from decide.scoring import score
        # approve: compare an expected-cost abstain rule with a plain confidence
        # rule, over abstain shares from 0 to (1 - coverage); keep the best q_appr.
        risk = np.minimum(3 * (1 - pa), pa)
        conf = np.maximum(pa, 1 - pa)
        best = None
        for mode in ("cost", "conf"):
            for qq in np.linspace(0.0, 1 - coverage, 9):
                c = dict(cfg, appr_mode=mode)
                if mode == "cost":
                    c["tau_appr"] = float(np.quantile(risk, 1 - qq)) if qq > 0 else 9.0
                else:
                    c["tau_appr"] = float(np.quantile(conf, qq)) if qq > 0 else -1.0
                _, _, d = score(records, Policy(c).responses(route_logits, appr_logits, gr_logits))
                if best is None or d["q_appr"] > best[0] + 1e-9:
                    best = (d["q_appr"], mode, qq, c)
        if verbose:
            print("  approve abstain rule: %s, abstain share %.3f, q_appr %.4f" % (best[1], best[2], best[0]))
        pol = Policy(best[3])
    return pol
