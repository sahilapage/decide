#!/usr/bin/env python3
"""
Self-check for the Decide service.

This is THE SAME SCORING CODE THE JUDGES RUN. The only differences are that it
defaults to your dev split instead of the hidden test set, and that it writes
results.json for you. There is no hidden extra criterion: what you see here is
what you are scored on.

Start your service, then:

    python selfcheck.py --url http://localhost:8000/v1/decide

It will print your gates, your four ranked scores, and the metrics behind them,
and write results.json in the format your submission needs.

Use --limit while you are iterating; run it over the full dev set before you
submit. Note that your own latency numbers depend on your machine - the judges
re-measure on theirs, so treat yours as a guide, not a guarantee.

Measurement rules, which are deliberate and which the judges also apply:

* One request at a time. No concurrency, no batching. A submission that is
  only fast in batch is not fast.
* A warm-up phase is discarded before any timing is kept.
* Decision types are interleaved in a fixed shuffled order.
* Latency is wall-clock around the HTTP call over a persistent connection,
  including serialisation. That is what a caller actually experiences.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
import http.client
import urllib.parse
from collections import Counter, defaultdict

QUEUES = ["billing", "technical", "account", "shipping", "general"]
DECISIONS = ["route", "approve", "grounded"]
FP_COST = 3.0          # wrongly auto-approving vs wrongly holding
LAT_FULL = 25.0        # ms for full latency marks
LAT_GATE = 100.0       # ms hard floor
TARGET_COVERAGE = 0.80
WEIGHTS = {"quality": 0.40, "calibration": 0.25, "latency": 0.20, "abstention": 0.15}


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

class Client:
    """
    Persistent HTTP/1.1 connection.

    A fresh TCP connection per request adds tens of milliseconds of setup on
    some platforms, which would swamp the thing we are trying to measure - on
    Windows loopback it put p95 at 24ms for a server doing 0.3ms of work. Real
    callers keep the connection alive, so the harness does too. If the server
    closes the connection or errors, we reconnect and carry on.
    """

    def __init__(self, url, timeout):
        u = urllib.parse.urlparse(url)
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.https else 80)
        self.path = u.path or "/"
        self.timeout = timeout
        self.conn = None
        self.reconnects = 0

    def _connect(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.conn = cls(self.host, self.port, timeout=self.timeout)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def call(self, payload):
        """Return (response_dict_or_None, elapsed_ms, error_string_or_None)."""
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)),
                   "Connection": "keep-alive"}
        for attempt in (0, 1):
            if self.conn is None:
                try:
                    self._connect()
                except Exception as e:
                    return None, 0.0, type(e).__name__
            t0 = time.perf_counter()
            try:
                self.conn.request("POST", self.path, body=body, headers=headers)
                r = self.conn.getresponse()
                raw = r.read()
                dt = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                self.close()
                self.reconnects += 1
                if attempt == 0:
                    continue                      # one retry on a dropped connection
                return None, (time.perf_counter() - t0) * 1000.0, type(e).__name__
            if r.status != 200:
                return None, dt, "HTTP %d" % r.status
            try:
                return json.loads(raw), dt, None
            except Exception:
                return None, dt, "invalid JSON"
        return None, 0.0, "unreachable"


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------

def validate(resp, decision, request_id):
    """Return a list of human-readable schema violations."""
    errs = []
    if not isinstance(resp, dict):
        return ["response is not an object"]

    if resp.get("request_id") != request_id:
        errs.append("request_id not echoed")
    if resp.get("decision") != decision:
        errs.append("decision field wrong or missing")

    ab = resp.get("abstain")
    if not isinstance(ab, bool):
        errs.append("abstain must be a boolean")

    lat = resp.get("latency_ms")
    if not isinstance(lat, (int, float)) or isinstance(lat, bool):
        errs.append("latency_ms must be a number")

    if decision == "route":
        if resp.get("type") != "choice":
            errs.append("type must be 'choice'")
        if resp.get("value") not in QUEUES:
            errs.append("value is not one of the five queues")
        probs = resp.get("probs")
        if not isinstance(probs, dict):
            errs.append("probs missing or not an object")
        else:
            missing = [q for q in QUEUES if q not in probs]
            if missing:
                errs.append("probs missing queues: %s" % ",".join(missing))
            else:
                try:
                    tot = sum(float(probs[q]) for q in QUEUES)
                    if abs(tot - 1.0) > 0.01:
                        errs.append("probs sum to %.4f, not 1.0" % tot)
                    if any(float(probs[q]) < 0 for q in QUEUES):
                        errs.append("negative probability")
                except (TypeError, ValueError):
                    errs.append("probs contains a non-number")
        c = resp.get("confidence")
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0.0 <= c <= 1.0:
            errs.append("confidence missing or outside [0,1]")
        elif isinstance(probs, dict) and resp.get("value") in probs:
            try:
                if abs(float(probs[resp["value"]]) - float(c)) > 0.01:
                    errs.append("confidence does not match probs[value]")
            except (TypeError, ValueError):
                pass
    else:
        if resp.get("type") != "binary":
            errs.append("type must be 'binary'")
        if not isinstance(resp.get("value"), bool):
            errs.append("value must be a boolean")
        p = resp.get("p_true")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
            errs.append("p_true missing or outside [0,1]")
    return errs


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def macro_f1(pairs, classes):
    """pairs: list of (truth, prediction)."""
    if not pairs:
        return 0.0
    out = []
    for c in classes:
        tp = sum(1 for y, p in pairs if y == c and p == c)
        fp = sum(1 for y, p in pairs if y != c and p == c)
        fn = sum(1 for y, p in pairs if y == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / len(out)


def brier(probs, ys):
    return sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(ys) if ys else 0.0


def ece(confs, correct, bins=15):
    """Expected calibration error. `correct` is 1/0 per prediction."""
    if not confs:
        return 0.0
    tot = 0.0
    n = len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs)
               if (lo < c <= hi) or (b == 0 and c <= 0.0)]
        if not idx:
            continue
        conf = sum(confs[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(conf - acc)
    return tot


def cost_error(ys, preds, fp_cost=FP_COST):
    """Mean cost. ys/preds are booleans; approving wrongly costs fp_cost."""
    if not ys:
        return 0.0
    c = 0.0
    for y, p in zip(ys, preds):
        if p and not y:
            c += fp_cost
        elif y and not p:
            c += 1.0
    return c / len(ys)


def selective_risk(confs, correct, coverage=TARGET_COVERAGE):
    """
    Error rate over the most-confident `coverage` fraction.

    This ignores the team's own abstain flag on purpose: it measures whether
    their confidence actually ranks their errors, which is comparable across
    submissions. Their declared coverage is reported separately.
    """
    if not confs:
        return 0.0
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    k = max(1, int(round(len(order) * coverage)))
    kept = order[:k]
    return 1.0 - sum(correct[i] for i in kept) / len(kept)


def skill(value, trivial, best=1.0):
    """Normalise a metric against the best trivial policy. 0 = no better."""
    if best == trivial:
        return 0.0
    return max(0.0, min(1.0, (value - trivial) / (best - trivial)))


COST_ANCHOR = 0.25


def multiclass_brier(prob_rows, truths, classes):
    """Mean squared error of the full predicted distribution."""
    if not truths:
        return 0.0
    tot = 0.0
    for pr, y in zip(prob_rows, truths):
        tot += sum((float(pr.get(c, 0.0)) - (1.0 if c == y else 0.0)) ** 2 for c in classes)
    return tot / len(truths)


def brier_skill(score, reference):
    """1.0 when perfect, 0.0 when no better than predicting the base rate."""
    if reference <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - score / reference))


def cost_score(cost, hold_cost, anchor=COST_ANCHOR):
    """
    Map a cost-weighted error onto [0, 1].

    Anchored rather than clipped at the trivial policy. Refusing every request
    ("always hold") costs the positive rate and earns `anchor`; zero cost earns
    1.0; the score decays to 0 as cost rises above the trivial bar. An earlier
    version floored at exactly the trivial cost, which meant a competent model
    that had simply left its threshold at 0.5 scored nothing at all, with no
    gradient to tell it apart from a model that was genuinely bad.
    """
    if hold_cost <= 0:
        # Every answered case was a hold, so "refuse everything" is free. Return
        # the anchor rather than a perfect score - otherwise answering only the
        # certain holds buys 1.0 on this axis.
        return anchor
    rel = (hold_cost - cost) / hold_cost        # +1 at zero cost, 0 at the bar
    return max(0.0, min(1.0, anchor + (1.0 - anchor) * rel))


# --------------------------------------------------------------------------

def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="full endpoint URL, e.g. http://host:8000/v1/decide")
    ap.add_argument("--eval", default="data/dev.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N records")
    ap.add_argument("--warmup", type=int, default=40, help="discarded requests before timing")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="results.json", help="where to write results")
    args = ap.parse_args()

    rows = load(args.eval)
    if args.limit:
        rows = rows[:args.limit]
    print("endpoint : %s" % args.url)
    print("eval set : %s  (%d records, %d calls)\n" % (args.eval, len(rows), len(rows) * 3))

    client = Client(args.url, args.timeout)

    # ---- warm up, discarded -------------------------------------------------
    print("warming up (%d calls, discarded) ..." % args.warmup, end=" ", flush=True)
    for i in range(args.warmup):
        r = rows[i % len(rows)]
        client.call({"request_id": r["request_id"],
                     "decision": DECISIONS[i % 3],
                     "state": r["state"]})
    print("done\n")

    # ---- interleaved sequential pass ---------------------------------------
    plan = [(r, d) for r in rows for d in DECISIONS]
    random.Random(args.seed).shuffle(plan)

    lat = defaultdict(list)
    self_reported = defaultdict(list)
    violations = Counter()
    n_bad = 0
    transport_errors = Counter()
    res = {d: [] for d in DECISIONS}          # (truth, resp)

    t_start = time.perf_counter()
    for i, (r, d) in enumerate(plan):
        if i % 500 == 0:
            print("  %5d / %d" % (i, len(plan)), flush=True)
        resp, ms, err = client.call({"request_id": r["request_id"],
                                     "decision": d,
                                     "state": r["state"]})
        lat[d].append(ms)
        if err:
            transport_errors[err] += 1
            n_bad += 1
            res[d].append((r["labels"][d], None))
            continue
        errs = validate(resp, d, r["request_id"])
        if errs:
            n_bad += 1
            for e in errs:
                violations[e] += 1
            res[d].append((r["labels"][d], None))
        else:
            res[d].append((r["labels"][d], resp))
            self_reported[d].append(float(resp["latency_ms"]))
    wall = time.perf_counter() - t_start
    print("  %5d / %d   (%.1fs)\n" % (len(plan), len(plan), wall))

    # ---- latency ------------------------------------------------------------
    per_p95 = {d: percentile(lat[d], 0.95) for d in DECISIONS}
    per_p50 = {d: percentile(lat[d], 0.50) for d in DECISIONS}
    all_lat = [x for d in DECISIONS for x in lat[d]]
    mean_p95 = sum(per_p95.values()) / 3.0

    print("=" * 66)
    print("LATENCY   (one request at a time, warm)")
    print("=" * 66)
    for d in DECISIONS:
        print("  %-9s p50 %7.2f ms    p95 %7.2f ms    n=%d"
              % (d, per_p50[d], per_p95[d], len(lat[d])))
    print("  %-9s p50 %7.2f ms    p95 %7.2f ms"
          % ("overall", percentile(all_lat, 0.50), percentile(all_lat, 0.95)))
    print("  mean of per-decision p95: %.2f ms   (gate %.0f, full marks %.0f)"
          % (mean_p95, LAT_GATE, LAT_FULL))
    sr = [x for d in DECISIONS for x in self_reported[d]]
    if sr:
        sr_p95 = percentile(sr, 0.95)
        print("  server self-reported p95: %.2f ms   (transport overhead %.2f ms)"
              % (sr_p95, percentile(all_lat, 0.95) - sr_p95))
        # Only flag a gap that is both proportionally and absolutely large.
        # A couple of milliseconds is ordinary framework overhead.
        gap = percentile(all_lat, 0.95) - sr_p95
        if sr_p95 > 0 and gap > 5.0 and percentile(all_lat, 0.95) > 3 * max(sr_p95, 0.5):
            print("      note: measured latency is far above what the server reports;")
            print("            check for per-request connection setup or blocking I/O")
    if client.reconnects:
        print("  connection drops: %d (server may not support keep-alive)" % client.reconnects)

    # ---- schema -------------------------------------------------------------
    print("\n" + "=" * 66)
    print("SCHEMA")
    print("=" * 66)
    total = len(plan)
    print("  valid responses : %d / %d  (%.2f%%)"
          % (total - n_bad, total, 100.0 * (total - n_bad) / total))
    if transport_errors:
        print("  transport errors:")
        for k, v in transport_errors.most_common():
            print("      %-28s %d" % (k, v))
    if violations:
        print("  schema violations:")
        for k, v in violations.most_common(12):
            print("      %-44s %d" % (k[:44], v))
    if not violations and not transport_errors:
        print("  no violations")

    # ---- quality ------------------------------------------------------------
    # Abstained cases are EXCLUDED from quality scoring, as the brief promises.
    #
    # On its own that is exploitable: abstain on everything hard and your
    # quality is computed over the easy remainder. So each decision's quality
    # is scaled by its coverage against the 80% target - answer less than that
    # and the score comes down proportionally. Calibration and abstention are
    # still measured over every record, because confidence on a case you
    # declined to answer is still a claim about what you know.
    report = {}

    def answered(resp):
        return bool(resp) and not resp["abstain"]

    # route
    #
    # Calibration and abstention are read from the PROBABILITIES, never from
    # the committed `value`. Otherwise a team can answer deliberately wrongly
    # on the cases it abstains on: quality ignores them, the reported
    # confidence is honest for that wrong answer, and `full_error` inflates, so
    # the abstention ratio improves. Measured at +1.8 points before this
    # change. `value` now affects quality only.
    pairs, confs, corr, ans = [], [], [], []
    prob_rows, truths = [], []
    for truth, resp in res["route"]:
        pred = resp["value"] if resp else None
        probs = resp["probs"] if resp else {}
        top = max(probs, key=probs.get) if probs else None
        confs.append(float(probs[top]) if top is not None else 0.0)
        corr.append(1.0 if top is not None and top == truth else 0.0)
        prob_rows.append(probs)
        truths.append(truth)
        ans.append(answered(resp))
        if answered(resp):
            pairs.append((truth, pred))
    route_cov = sum(ans) / len(ans)
    route_f1 = macro_f1(pairs, QUEUES)
    route_acc = sum(corr) / len(corr)
    route_ece = ece(confs, corr)
    route_risk = selective_risk(confs, corr)
    route_declared = [1.0 if a else 0.0 for a in ans]

    # approve
    ys = [1.0 if t else 0.0 for t, _ in res["approve"]]
    preds, aprobs, acorr, aans, apairs = [], [], [], [], []
    for truth, resp in res["approve"]:
        p = float(resp["p_true"]) if resp else 0.0
        v = bool(resp["value"]) if resp else False
        preds.append(v)
        aprobs.append(p)
        # Confidence is the probability of the value actually returned, not of
        # the more likely class. Those differ whenever a team uses a threshold
        # other than 0.5 - which the brief tells them to do here.
        acorr.append(1.0 if resp and v == bool(truth) else 0.0)
        aans.append(answered(resp))
        if answered(resp):
            apairs.append((bool(truth), v))
    # Binary calibration is measured on the probability against the outcome,
    # which is what "calibrated" means and is independent of the value chosen.
    aconf = list(aprobs)
    acorr = [y for y in ys]
    appr_cov = sum(aans) / len(aans)
    appr_cost = cost_error([bool(y) for y, a in zip(ys, aans) if a],
                           [p for p, a in zip(preds, aans) if a])
    appr_f1 = macro_f1(apairs, [True, False])
    appr_brier = brier(aprobs, ys)
    appr_ece = ece(aconf, acorr)
    # ECE wants (probability, outcome); abstention and the full-coverage error
    # want (confidence, was-it-right). Reusing the calibration lists for both
    # made the abstention axis measure the negative-class rate instead of the
    # error rate, which capped a perfect model at 0.53 on that axis. Rebuilt
    # here from the probability only, so the M1 guarantee still holds: nothing
    # downstream reads the committed value.
    aconf = [max(p, 1.0 - p) for p in aprobs]
    acorr = [1.0 if (p > 0.5) == (y > 0.5) else 0.0 for p, y in zip(aprobs, ys)]
    appr_risk = selective_risk(aconf, acorr)

    # grounded
    gys = [1.0 if t else 0.0 for t, _ in res["grounded"]]
    gpreds, gprobs, gcorr, gans, gpairs = [], [], [], [], []
    for truth, resp in res["grounded"]:
        p = float(resp["p_true"]) if resp else 0.0
        v = bool(resp["value"]) if resp else False
        gpreds.append(v)
        gprobs.append(p)
        gcorr.append(1.0 if resp and v == bool(truth) else 0.0)
        gans.append(answered(resp))
        if answered(resp):
            gpairs.append((bool(truth), v))
    gconf = list(gprobs)
    gcorr = [y for y in gys]
    gr_cov = sum(gans) / len(gans)
    gr_f1 = macro_f1(gpairs, [True, False])
    gr_brier = brier(gprobs, gys)
    gr_ece = ece(gconf, gcorr)
    gconf = [max(p, 1.0 - p) for p in gprobs]
    gcorr = [1.0 if (p > 0.5) == (y > 0.5) else 0.0 for p, y in zip(gprobs, gys)]
    gr_risk = selective_risk(gconf, gcorr)

    # ---- trivial references, computed on the SAME answered subsets ----------
    # Comparing an answered-subset score against a full-set trivial policy
    # would flatter anyone who abstained on the hard cases.
    ans_route = [t for (t, _), a in zip(res["route"], ans) if a]
    if ans_route:
        maj_route = Counter(ans_route).most_common(1)[0][0]
        triv_route_f1 = macro_f1([(t, maj_route) for t in ans_route], QUEUES)
    else:
        triv_route_f1 = 0.0

    # The trivial reference is computed on the SAME answered subset the cost is
    # measured on. Taking it from the full set instead looks safer but is not:
    # cost then falls as you abstain on hard cases while the bar stays put, and
    # heavy abstention beats answering honestly. The degenerate case where the
    # answered set holds no positives is handled inside cost_score, which
    # returns the anchor rather than a free 1.0.
    ans_ys = [y for y, a in zip(ys, aans) if a]
    rate = sum(ans_ys) / len(ans_ys) if ans_ys else 0.0
    hold_cost = rate * 1.0                                  # refuse everything
    approve_cost = (1 - rate) * FP_COST                     # approve everything
    triv_cost = min(hold_cost, approve_cost)

    ans_gys = [y for y, a in zip(gys, gans) if a]
    if ans_gys:
        triv_gr_f1 = max(
            macro_f1([(bool(y), True) for y in ans_gys], [True, False]),
            macro_f1([(bool(y), False) for y in ans_gys], [True, False]))
    else:
        triv_gr_f1 = 0.0

    print("\n" + "=" * 66)
    print("QUALITY")
    print("=" * 66)
    print("  route      macro-F1 %.4f   acc %.4f        (majority %.4f)"
          % (route_f1, route_acc, triv_route_f1))
    print("  approve    cost     %.4f   macro-F1 %.4f   (%.0f:1 asymmetry)"
          % (appr_cost, appr_f1, FP_COST))
    print("             trivial policies: always-hold %.4f   always-approve %.4f"
          % (hold_cost, approve_cost))
    print("  grounded   macro-F1 %.4f                   (majority %.4f)"
          % (gr_f1, triv_gr_f1))
    print("  scored on answered cases only; coverage "
          "route %.3f  approve %.3f  grounded %.3f" % (route_cov, appr_cov, gr_cov))
    if min(route_cov, appr_cov, gr_cov) < TARGET_COVERAGE:
        print("             quality is scaled down where coverage is below %.0f%%"
              % (TARGET_COVERAGE * 100))

    print("\n" + "=" * 66)
    print("CALIBRATION")
    print("=" * 66)
    print("  route      ECE %.4f" % route_ece)
    print("  approve    ECE %.4f   Brier %.4f" % (appr_ece, appr_brier))
    print("  grounded   ECE %.4f   Brier %.4f" % (gr_ece, gr_brier))
    mean_ece = (route_ece + appr_ece + gr_ece) / 3.0

    # Brier skill against predicting the base rate. ECE alone rewards a model
    # that says nothing: always emitting the prior scores near-zero ECE. Brier
    # punishes exactly that, so the two together ask for probabilities that are
    # both honest and informative.
    route_prior = {c: sum(1 for t in truths if t == c) / len(truths) for c in QUEUES}
    route_brier = multiclass_brier(prob_rows, truths, QUEUES)
    route_ref = multiclass_brier([route_prior] * len(truths), truths, QUEUES)
    arate = sum(ys) / len(ys)
    grate_ = sum(gys) / len(gys)
    bs = [brier_skill(route_brier, route_ref),
          brier_skill(appr_brier, arate * (1 - arate)),
          brier_skill(gr_brier, grate_ * (1 - grate_))]
    mean_skill = sum(bs) / 3.0
    mean_brier = (appr_brier + gr_brier) / 2.0
    print("  mean ECE %.4f    Brier skill %.4f  (route %.3f / approve %.3f / grounded %.3f)"
          % (mean_ece, mean_skill, bs[0], bs[1], bs[2]))

    print("\n" + "=" * 66)
    print("ABSTENTION")
    print("=" * 66)
    cov = sum(route_declared) / len(route_declared)
    print("  risk @ %.0f%% coverage, by the team's own confidence ranking:"
          % (TARGET_COVERAGE * 100))
    print("      route %.4f    approve %.4f    grounded %.4f"
          % (route_risk, appr_risk, gr_risk))
    full_err = [1 - route_acc,
                1 - sum(acorr) / len(acorr),
                1 - sum(gcorr) / len(gcorr)]
    print("  error rate at full coverage:")
    print("      route %.4f    approve %.4f    grounded %.4f" % tuple(full_err))
    dec_cov = {d: sum(1 for _, r in res[d] if r and not r["abstain"]) / len(res[d])
               for d in DECISIONS}
    print("  declared coverage (their own abstain flag):")
    print("      route %.3f    approve %.3f    grounded %.3f"
          % (dec_cov["route"], dec_cov["approve"], dec_cov["grounded"]))
    if min(dec_cov.values()) < TARGET_COVERAGE:
        print("      note: coverage below %.0f%% on at least one decision, so quality"
              % (TARGET_COVERAGE * 100))
        print("            is scaled down there. Above the target costs nothing.")

    # ---- gates --------------------------------------------------------------
    gates = {
        "schema_clean": n_bad == 0,
        "latency_under_gate": mean_p95 <= LAT_GATE,
        "reachable": sum(transport_errors.values()) == 0,
    }

    # ---- ranked score -------------------------------------------------------
    def cov_scale(cov):
        """Answer less than the 80% target and quality comes down with it."""
        return min(1.0, cov / TARGET_COVERAGE) if TARGET_COVERAGE > 0 else 1.0

    q_route = skill(route_f1, triv_route_f1) * cov_scale(route_cov)
    q_appr = cost_score(appr_cost, hold_cost) * cov_scale(appr_cov)
    q_gr = skill(gr_f1, triv_gr_f1) * cov_scale(gr_cov)
    quality = (q_route + q_appr + q_gr) / 3.0

    # Half honesty (ECE), half informativeness (Brier skill).
    calibration = 0.5 * max(0.0, 1.0 - mean_ece / 0.20) + 0.5 * mean_skill

    if mean_p95 <= LAT_FULL:
        latency = 1.0
    elif mean_p95 >= LAT_GATE:
        latency = 0.0
    else:
        latency = (LAT_GATE - mean_p95) / (LAT_GATE - LAT_FULL)

    ab = []
    for risk, err in zip([route_risk, appr_risk, gr_risk], full_err):
        ab.append(max(0.0, 1.0 - risk / err) if err > 1e-9 else 1.0)
    abstention = sum(ab) / 3.0

    parts = {"quality": quality, "calibration": calibration,
             "latency": latency, "abstention": abstention}
    final = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) * 100

    print("\n" + "=" * 66)
    print("GATES")
    print("=" * 66)
    for k, v in gates.items():
        print("  %-22s %s" % (k, "PASS" if v else "FAIL"))
    if not all(gates.values()):
        print("\n  *** NOT RANKED - a gate failed ***")

    print("\n" + "=" * 66)
    print("SCORE")
    print("=" * 66)
    for k in ("quality", "calibration", "latency", "abstention"):
        print("  %-12s %.4f  x %2.0f%%  = %6.2f"
              % (k, parts[k], WEIGHTS[k] * 100, WEIGHTS[k] * parts[k] * 100))
    print("  %-12s %28.2f / 100" % ("TOTAL", final))
    print("  tie-breakers: p95 %.2f ms" % mean_p95)

    report = {
        "url": args.url,
        "eval": args.eval,
        "n_records": len(rows),
        "n_calls": total,
        "gates": gates,
        "schema": {"valid": total - n_bad, "total": total,
                   "violations": dict(violations),
                   "transport_errors": dict(transport_errors)},
        "latency_ms": {"p50": per_p50, "p95": per_p95, "mean_p95": mean_p95,
                       "overall_p50": percentile(all_lat, 0.50),
                       "overall_p95": percentile(all_lat, 0.95),
                       "server_self_reported_p95": percentile(sr, 0.95) if sr else None,
                       "reconnects": client.reconnects},
        "quality": {"route_macro_f1": route_f1, "route_acc": route_acc,
                    "coverage": {"route": route_cov, "approve": appr_cov,
                                 "grounded": gr_cov},
                    "approve_cost": appr_cost, "approve_macro_f1": appr_f1,
                    "grounded_macro_f1": gr_f1,
                    "fp_cost": FP_COST,
                    "trivial": {"route_macro_f1": triv_route_f1,
                                "approve_cost_always_hold": hold_cost,
                                "approve_cost_always_approve": approve_cost,
                                "grounded_macro_f1": triv_gr_f1}},
        "calibration": {"route_ece": route_ece, "approve_ece": appr_ece,
                        "grounded_ece": gr_ece, "approve_brier": appr_brier,
                        "grounded_brier": gr_brier, "mean_ece": mean_ece,
                        "mean_brier": mean_brier, "brier_skill": mean_skill,
                        "route_brier": route_brier},
        "abstention": {"risk_at_80": {"route": route_risk, "approve": appr_risk,
                                      "grounded": gr_risk},
                       "full_error": dict(zip(DECISIONS, full_err)),
                       "declared_coverage": dec_cov},
        "score_parts": parts,
        "score": final,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s  (include this in your submission)" % args.json)

    print("""
Reading this report
-------------------
  quality      how often you are right on the cases you CHOSE to answer,
               scored against the best trivial policy for each decision, then
               scaled by coverage against an 80%% target
  calibration  half ECE (is 0.8 really 0.8?) and half Brier skill (do your
               probabilities beat quoting the base rate?). Honest but
               uninformative scores no better than informative but wrong
  latency      p95 per decision type, averaged. Full marks at 25ms, nothing
               at the 100ms gate
  abstention   whether your confidence actually ranks your own errors. If
               risk@80%% is close to your full error rate, it does not

Calibration and abstention are read from your PROBABILITIES, never from the
value you commit to. Returning an answer you do not believe gains nothing.

The trivial-policy lines under QUALITY are the bar. If a decision scores near
zero, compare it to them before assuming the model is the problem.""")

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
