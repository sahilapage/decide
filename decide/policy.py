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
        mode = self.cfg.get("appr_mode", "cost")
        if mode == "conf":
            return value, p, max(p, 1.0 - p) < self.cfg["tau_appr"]
        if mode == "ratio":
            # marginal effect of answering this case on 1 - cost/hold_cost
            s = min(3.0 * (1.0 - p), p) - self.cfg["lam_appr"] * p
            return value, p, s > self.cfg["tau_appr"]
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
        cfg = tune_abstention(cfg, records, route_logits, appr_logits, gr_logits, coverage, verbose)
        pol = Policy(cfg)
    return pol


def tune_abstention(cfg, records, route_logits, appr_logits, gr_logits,
                    coverage=TARGET_COVERAGE, verbose=True, tol=0.002):
    """
    Pick each decision's abstain rule against the offline scorer.

    Abstaining only changes that decision's quality term, and only helps while
    coverage stays above the 80% target. For each decision we search abstain
    shares in [0, 1 - coverage] and keep the SMALLEST share whose quality is
    within `tol` of the best: no abstaining that buys nothing.
    """
    from decide.scoring import score

    route_logits = np.asarray(route_logits, dtype=np.float64)
    pr = softmax(route_logits / cfg["T_route"]).max(1)
    pa = sigmoid(cfg["a_appr"] * np.asarray(appr_logits) + cfg["b_appr"])
    pg = sigmoid(cfg["a_gr"] * np.asarray(gr_logits) + cfg["b_gr"])
    shares = np.linspace(0.0, 1 - coverage, 13)
    cfg = dict(cfg)

    def pick(key, trials):
        res = []
        for share, c in trials:
            _, _, d = score(records, Policy(c).responses(route_logits, appr_logits, gr_logits))
            res.append((d[key], share, c))
        top = max(r[0] for r in res)
        ok = [r for r in res if r[0] >= top - tol]
        return min(ok, key=lambda r: r[1])

    # route: abstain on the lowest top-class probability
    q_r, s_r, c_r = pick("q_route", [(s, dict(cfg, tau_route=float(np.quantile(pr, s)) if s > 0 else -1.0))
                                     for s in shares])
    cfg["tau_route"] = c_r["tau_route"]
    # grounded: abstain on the least decisive probabilities
    cg = np.maximum(pg, 1 - pg)
    q_g, s_g, c_g = pick("q_gr", [(s, dict(cfg, tau_gr=float(np.quantile(cg, s)) if s > 0 else -1.0))
                                  for s in shares])
    cfg["tau_gr"] = c_g["tau_gr"]
    # approve: expected-cost rule, confidence rule, or the ratio rule
    risk = np.minimum(3 * (1 - pa), pa)
    conf = np.maximum(pa, 1 - pa)
    trials = []
    for mode, lam in [("cost", 0.0), ("conf", 0.0)] + [("ratio", l) for l in np.linspace(0.0, 2.0, 21)]:
        for s in shares:
            c = dict(cfg, appr_mode=mode, lam_appr=float(lam))
            if mode == "cost":
                c["tau_appr"] = float(np.quantile(risk, 1 - s)) if s > 0 else 9.0
            elif mode == "conf":
                c["tau_appr"] = float(np.quantile(conf, s)) if s > 0 else -1.0
            else:
                c["tau_appr"] = float(np.quantile(risk - lam * pa, 1 - s)) if s > 0 else 9.0
            trials.append((s, c))
    q_a, s_a, c_a = pick("q_appr", trials)
    cfg.update({k: c_a[k] for k in ("appr_mode", "lam_appr", "tau_appr")})
    if verbose:
        print("  abstain shares: route %.3f (q %.4f)  approve %.3f %s lam %.2f (q %.4f)  grounded %.3f (q %.4f)"
              % (s_r, q_r, s_a, cfg["appr_mode"], cfg["lam_appr"], q_a, s_g, q_g))
    return cfg
