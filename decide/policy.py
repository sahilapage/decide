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
TARGET_COVERAGE = 0.85  # abstain at most 15%: margin above the 80% target on unseen data


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


A_MAX, B_MAX = 1.5, 5.0   # bounds on binary calibration slope / offset


def fit_route_calibration(logits: np.ndarray, y: np.ndarray):
    """
    p = (1 - eps) * softmax(z / T + bias) + eps / 5

    Temperature alone cannot express what the routing labels look like: about
    5-6% of tickets carry a label unrelated to their text, so even the clearest
    ticket is right only ~95% of the time. `eps` is that floor (a uniform
    label-noise mixture), `bias` absorbs per-queue prior shift. Fitted by NLL.
    """
    from scipy.optimize import minimize

    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    n, k = z.shape

    def unpack(w):
        T = math.exp(w[0])
        bias = np.concatenate([[0.0], w[1:k]])
        eps = 1.0 / (1.0 + math.exp(-w[k]))
        return T, bias, eps

    def nll(w):
        T, bias, eps = unpack(w)
        p = (1 - eps) * softmax(z / T + bias) + eps / k
        return float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1)).mean())

    T0 = fit_temperature_multiclass(z, y)
    w0 = np.concatenate([[math.log(T0)], np.zeros(k - 1), [-3.0]])
    r = minimize(nll, w0, method="L-BFGS-B",
                 bounds=[(math.log(0.05), math.log(20))] + [(-3, 3)] * (k - 1) + [(-9, 0)])
    T, bias, eps = unpack(r.x)
    return float(T), [float(b) for b in bias], float(eps)


def fit_binary_calibration(logit: np.ndarray, y: np.ndarray):
    """
    p = lo + (hi - lo) * sigmoid(a * z + b)

    Platt scaling with a learned floor and ceiling. For approve, ~20% of the
    requests the rule would allow are refused for account-history reasons the
    ticket never shows, so P(yes) saturates well below 1; plain Platt has to
    bend its slope to fake that plateau.
    """
    from scipy.optimize import minimize

    x = np.asarray(logit, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    a0, b0 = fit_platt(x, yy)

    def nll(w):
        a, b, lo, hi = w
        p = lo + (hi - lo) * sigmoid(a * x + b)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())

    # Bounded: on a nearly separable decision (grounded makes ~1 error in 3,000)
    # the unconstrained fit runs the slope/offset off to infinity and saturates
    # every probability at exactly 0 or 1.
    a0 = min(max(a0, 0.05), A_MAX)
    b0 = min(max(b0, -B_MAX), B_MAX)
    r = minimize(nll, np.array([a0, b0, 0.0, 1.0]), method="L-BFGS-B",
                 bounds=[(0.05, A_MAX), (-B_MAX, B_MAX), (0.0, 0.3), (0.7, 1.0)])
    return [float(v) for v in r.x]


class Policy:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    # ---- calibrated probabilities ------------------------------------------
    def route_probs(self, logits):
        z = np.asarray(logits, dtype=np.float64)
        bias = np.asarray(self.cfg.get("bias_route", [0.0] * 5))
        eps = self.cfg.get("eps_route", 0.0)
        return (1 - eps) * softmax(z / self.cfg["T_route"] + bias) + eps / 5

    def p_approve(self, logit):
        c = self.cfg
        lo, hi = c.get("lo_appr", 0.0), c.get("hi_appr", 1.0)
        return lo + (hi - lo) * sigmoid(c["a_appr"] * np.asarray(logit, dtype=np.float64) + c["b_appr"])

    def p_grounded(self, logit):
        c = self.cfg
        lo, hi = c.get("lo_gr", 0.0), c.get("hi_gr", 1.0)
        return lo + (hi - lo) * sigmoid(c["a_gr"] * np.asarray(logit, dtype=np.float64) + c["b_gr"])

    # ---- apply ------------------------------------------------------------
    def route(self, logits: np.ndarray):
        probs = self.route_probs(logits)
        k = int(probs.argmax())
        conf = float(probs[k])
        pd = {q: float(p) for q, p in zip(QUEUES, probs)}
        return QUEUES[k], pd, conf, conf < self.cfg["tau_route"]

    def approve(self, logit: float):
        p = float(self.p_approve(logit))
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
        p = float(self.p_grounded(logit))
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


def refit_scales(struct_cfg, route_logits, y_route, appr_logits, y_appr, gr_logits, y_gr):
    """
    Keep the structural noise parameters of `struct_cfg` (route eps + bias,
    binary floor/ceiling: estimated on 6,500 out-of-fold rows) and refit only
    the scale parameters (T, a, b) on the deployed model's own logits.
    """
    from scipy.optimize import minimize

    cfg = dict(struct_cfg)
    z = np.asarray(route_logits, dtype=np.float64)
    y = np.asarray(y_route)
    bias = np.asarray(cfg.get("bias_route", [0.0] * 5))
    eps = cfg.get("eps_route", 0.0)

    def nll_T(w):
        p = (1 - eps) * softmax(z / math.exp(w[0]) + bias) + eps / 5
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    r = minimize(nll_T, np.array([math.log(cfg["T_route"])]), method="L-BFGS-B",
                 bounds=[(math.log(0.05), math.log(20))])
    cfg["T_route"] = float(math.exp(r.x[0]))
    for key, lg, yy in (("appr", appr_logits, y_appr), ("gr", gr_logits, y_gr)):
        x = np.asarray(lg, dtype=np.float64)
        t = np.asarray(yy, dtype=np.float64)
        lo, hi = cfg.get("lo_" + key, 0.0), cfg.get("hi_" + key, 1.0)

        def nll(w):
            p = np.clip(lo + (hi - lo) * sigmoid(w[0] * x + w[1]), 1e-9, 1 - 1e-9)
            return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())

        w0 = np.array([min(max(cfg["a_" + key], 0.05), A_MAX), min(max(cfg["b_" + key], -B_MAX), B_MAX)])
        r = minimize(nll, w0, method="L-BFGS-B", bounds=[(0.05, A_MAX), (-B_MAX, B_MAX)])
        cfg["a_" + key], cfg["b_" + key] = float(r.x[0]), float(r.x[1])
    return cfg


def fit_policy(route_logits, y_route, appr_logits, y_appr, gr_logits, y_gr,
               records=None, coverage=TARGET_COVERAGE, verbose=True, calib="noise",
               struct_cfg=None) -> Policy:
    """
    Fit calibration, then (if `records` are given) tune abstention against the
    offline scorer. calib="noise" uses the label-noise-aware calibrators
    (route: T + bias + uniform floor; binaries: Platt with floor/ceiling);
    calib="basic" is plain temperature / Platt.
    """
    route_logits = np.asarray(route_logits, dtype=np.float64)
    y_route = np.asarray(y_route)
    if struct_cfg is not None:
        cfg = refit_scales(struct_cfg, route_logits, y_route, appr_logits, y_appr, gr_logits, y_gr)
        cfg.update({"tau_route": -1.0, "tau_appr": 9.0, "tau_gr": -1.0, "appr_mode": "cost", "lam_appr": 0.0})
        if records is not None:
            cfg = tune_abstention(cfg, records, route_logits, appr_logits, gr_logits, coverage, verbose)
        return Policy(cfg)
    if calib == "noise":
        T, bias, eps = fit_route_calibration(route_logits, y_route)
        a_ap, b_ap, lo_ap, hi_ap = fit_binary_calibration(np.asarray(appr_logits), np.asarray(y_appr))
        a_gr, b_gr, lo_gr, hi_gr = fit_binary_calibration(np.asarray(gr_logits), np.asarray(y_gr))
    else:
        T, bias, eps = fit_temperature_multiclass(route_logits, y_route), [0.0] * 5, 0.0
        a_ap, b_ap = fit_platt(np.asarray(appr_logits), np.asarray(y_appr))
        a_gr, b_gr = fit_platt(np.asarray(gr_logits), np.asarray(y_gr))
        lo_ap, hi_ap, lo_gr, hi_gr = 0.0, 1.0, 0.0, 1.0
    cfg = {
        "T_route": T, "bias_route": bias, "eps_route": eps,
        "a_appr": a_ap, "b_appr": b_ap, "lo_appr": lo_ap, "hi_appr": hi_ap,
        "a_gr": a_gr, "b_gr": b_gr, "lo_gr": lo_gr, "hi_gr": hi_gr,
        "thr_appr": 0.75, "thr_gr": 0.5,
        "tau_route": -1.0, "tau_appr": 9.0, "tau_gr": -1.0, "appr_mode": "cost", "lam_appr": 0.0,
    }
    if records is not None:
        cfg = tune_abstention(cfg, records, route_logits, appr_logits, gr_logits, coverage, verbose)
    return Policy(cfg)


# ---------------------------------------------------------------- fast quality terms
# Vectorised copies of the selfcheck.py quality formulas, used only to search
# abstain thresholds quickly. decide/scoring.py stays the reference.
def _macro_f1(y, p, classes):
    out = []
    for c in classes:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def _skill(v, triv):
    return 0.0 if triv == 1.0 else max(0.0, min(1.0, (v - triv) / (1.0 - triv)))


def _cov_scale(cov):
    return min(1.0, cov / 0.80)


def q_route_fast(y, pred, answered):
    ya, pa = y[answered], pred[answered]
    if len(ya) == 0:
        return 0.0
    maj = np.bincount(ya, minlength=5).argmax()
    f1 = _macro_f1(ya, pa, range(5))
    triv = _macro_f1(ya, np.full_like(ya, maj), range(5))
    return _skill(f1, triv) * _cov_scale(answered.mean())


def q_appr_fast(y, value, answered, anchor=0.25):
    ya, va = y[answered].astype(bool), value[answered].astype(bool)
    if len(ya) == 0:
        return 0.0
    cost = (3.0 * np.sum(va & ~ya) + np.sum(~va & ya)) / len(ya)
    hold = ya.mean()
    if hold <= 0:
        return anchor * _cov_scale(answered.mean())
    rel = (hold - cost) / hold
    return max(0.0, min(1.0, anchor + (1 - anchor) * rel)) * _cov_scale(answered.mean())


def q_gr_fast(y, value, answered):
    ya, va = y[answered].astype(bool), value[answered].astype(bool)
    if len(ya) == 0:
        return 0.0
    f1 = _macro_f1(ya, va, [True, False])
    triv = max(_macro_f1(ya, np.ones_like(ya), [True, False]), _macro_f1(ya, np.zeros_like(ya), [True, False]))
    return _skill(f1, triv) * _cov_scale(answered.mean())


def tune_abstention(cfg, records, route_logits, appr_logits, gr_logits,
                    coverage=TARGET_COVERAGE, verbose=True, tol=0.002):
    """
    Pick each decision's abstain rule against the scorer's quality terms.

    Abstaining only changes that decision's quality term, and only helps while
    coverage stays above the 80% target. For each decision we search abstain
    shares in [0, 1 - coverage] and keep the SMALLEST share whose quality is
    within `tol` of the best: no abstaining that buys nothing.
    """
    route_logits = np.asarray(route_logits, dtype=np.float64)
    base = Policy(cfg)
    P = base.route_probs(route_logits)
    pr, pred = P.max(1), P.argmax(1)
    pa = base.p_approve(appr_logits)
    pg = base.p_grounded(gr_logits)
    QI = {q: i for i, q in enumerate(QUEUES)}
    yr = np.array([QI[r["labels"]["route"]] for r in records])
    ya = np.array([bool(r["labels"]["approve"]) for r in records])
    yg = np.array([bool(r["labels"]["grounded"]) for r in records])
    shares = np.linspace(0.0, 1 - coverage, 13)
    cfg = dict(cfg)

    def pick(res):
        top = max(r[0] for r in res)
        return min([r for r in res if r[0] >= top - tol], key=lambda r: r[1])

    res = []
    for s_ in shares:
        tau = float(np.quantile(pr, s_)) if s_ > 0 else -1.0
        res.append((q_route_fast(yr, pred, pr >= tau), s_, tau))
    q_r, s_r, cfg["tau_route"] = pick(res)

    cg = np.maximum(pg, 1 - pg)
    vg = pg > cfg["thr_gr"]
    res = []
    for s_ in shares:
        tau = float(np.quantile(cg, s_)) if s_ > 0 else -1.0
        res.append((q_gr_fast(yg, vg, cg >= tau), s_, tau))
    q_g, s_g, cfg["tau_gr"] = pick(res)

    va = pa > cfg["thr_appr"]
    risk = np.minimum(3 * (1 - pa), pa)
    conf = np.maximum(pa, 1 - pa)
    res = []
    for mode, lam in [("cost", 0.0), ("conf", 0.0)] + [("ratio", l) for l in np.linspace(0.0, 2.0, 21)]:
        for s_ in shares:
            if mode == "conf":
                tau = float(np.quantile(conf, s_)) if s_ > 0 else -1.0
                ans = conf >= tau
            else:
                score_ = risk if mode == "cost" else risk - lam * pa
                tau = float(np.quantile(score_, 1 - s_)) if s_ > 0 else 9.0
                ans = score_ <= tau
            res.append((q_appr_fast(ya, va, ans), s_, (mode, float(lam), tau)))
    q_a, s_a, (cfg["appr_mode"], cfg["lam_appr"], cfg["tau_appr"]) = pick(res)
    if verbose:
        print("  abstain shares: route %.3f (q %.4f)  approve %.3f %s lam %.2f (q %.4f)  grounded %.3f (q %.4f)"
              % (s_r, q_r, s_a, cfg["appr_mode"], cfg["lam_appr"], q_a, s_g, q_g))
    return cfg
