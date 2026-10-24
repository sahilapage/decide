"""
Run the shipped model over a whole dataset, in-process (no server, no port), and
print every answer plus the timing, the accuracy and the judges' score.

  python test.py                         # data/test.jsonl if present, else data/dev.jsonl
  python test.py path/to/any.jsonl       # swap in any dataset in the same format
  python test.py data/dev.jsonl --limit 20 --quiet

Dataset format: one JSON object per line (or a JSON list), each
  {"request_id": ..., "state": {subject, body, customer_tier, amount, draft_reply, kb_article},
   "labels": {"route": ..., "approve": ..., "grounded": ...}}      <- labels optional

Every record is asked all three decisions, exactly as the judges' harness does
(one call per decision). Times are per call: input cleaning, tokenising, one
ONNX forward pass of the shared encoder and the calibration/abstain policy.
With labels, the score uses the same metric code as selfcheck.py.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from app import State  # noqa: E402  same input validation as the API (nulls, extra fields)
from model import DecisionModel  # noqa: E402

DECISIONS = ("route", "approve", "grounded")
TTY = sys.stdout.isatty()


def c(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if TTY else text


def load(path):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def ask(model, decision, state):
    """One API-equivalent call. Returns (response dict shaped like the API body, ms)."""
    t0 = time.perf_counter()
    if decision == "route":
        value, probs, conf, abstain = model.route(state)
        resp = {"type": "choice", "value": value, "probs": {q: float(p) for q, p in probs.items()},
                "confidence": float(conf), "abstain": bool(abstain)}
    else:
        value, p, abstain = getattr(model, decision)(state)
        resp = {"type": "binary", "value": bool(value), "p_true": float(p), "abstain": bool(abstain)}
    return resp, (time.perf_counter() - t0) * 1000.0


def show(resp, truth):
    if resp["type"] == "choice":
        txt = "%-9s %.2f" % (resp["value"], resp["confidence"])
    else:
        txt = "%-9s %.2f" % ("yes" if resp["value"] else "no", resp["p_true"])
    txt += " " + (c("ABSTAIN", "33") if resp["abstain"] else "       ")
    if truth is None:
        return txt
    ok = resp["value"] == truth
    return txt + " " + (c("ok", "32") if ok else c("XX", "31;1"))


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = "data/test.jsonl" if os.path.exists(os.path.join(HERE, "data/test.jsonl")) else "data/dev.jsonl"
    ap.add_argument("dataset", nargs="?", default=default)
    ap.add_argument("--artifacts", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N records")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-record lines")
    ap.add_argument("--seed", type=int, default=7, help="call order, same default as selfcheck.py")
    args = ap.parse_args()

    recs = load(args.dataset)
    if args.limit:
        recs = recs[: args.limit]
    has_labels = all("labels" in r for r in recs)

    t_load = time.perf_counter()
    model = DecisionModel(args.artifacts)
    print("model: %s backend, loaded in %.1f s | dataset: %s (%d records, %d calls)%s\n" % (
        model.backend, time.perf_counter() - t_load, args.dataset, len(recs), 3 * len(recs),
        "" if has_labels else " | no labels: answers and timing only"))

    if not args.quiet:
        print(c("%-5s %-12s | %-26s | %-26s | %-26s | %s" % (
            "#", "request_id", "route", "approve", "grounded", "ms (3 calls)"), "1"))

    # Calls go out in the judges' order: every (record, decision) pair, shuffled
    # with selfcheck.py's seed. The abstention metric sorts by confidence, so the
    # order of tied confidences matters; this makes the score match theirs.
    plan = [(i, d) for i in range(len(recs)) for d in DECISIONS]
    random.Random(args.seed).shuffle(plan)
    states = [State(**(r.get("state") or {})).model_dump() for r in recs]
    got, times = {}, {d: [] for d in DECISIONS}
    t_all = time.perf_counter()
    for i, d in plan:
        got[i, d] = ask(model, d, states[i])
        times[d].append(got[i, d][1])
    wall = time.perf_counter() - t_all
    order = {d: [i for i, dd in plan if dd == d] for d in DECISIONS}
    outs = {d: [got[i, d][0] for i in order[d]] for d in DECISIONS}

    if not args.quiet:
        for i, rec in enumerate(recs):
            cells = [show(got[i, d][0], rec["labels"][d] if has_labels else None) for d in DECISIONS]
            ms = sum(got[i, d][1] for d in DECISIONS)
            print("%-5d %-12s | %s | %s | %s | %6.1f" % (i + 1, str(rec.get("request_id", ""))[:12], *cells, ms))

    # ---- summary ------------------------------------------------------------
    print("\n" + c("TIMING (per call, in-process: clean + tokenise + ONNX + policy)", "1"))
    allt = [t for d in DECISIONS for t in times[d]]
    for d in DECISIONS:
        print("  %-9s avg %6.2f ms   p50 %6.2f   p95 %6.2f   max %6.2f" % (
            d, np.mean(times[d]), pct(times[d], 50), pct(times[d], 95), max(times[d])))
    print("  %-9s avg %6.2f ms   p50 %6.2f   p95 %6.2f   | %d calls in %.1f s (%.0f calls/s)" % (
        "all", np.mean(allt), pct(allt, 50), pct(allt, 95), len(allt), wall, len(allt) / wall))

    abst = {d: np.mean([o["abstain"] for o in outs[d]]) for d in DECISIONS}
    if not has_labels:
        print("\n  abstain rate: " + "   ".join("%s %.1f%%" % (d, 100 * abst[d]) for d in DECISIONS))
        return

    print("\n" + c("ACCURACY", "1"))
    for d in DECISIONS:
        truth = [recs[i]["labels"][d] for i in order[d]]
        right = [o["value"] == t for o, t in zip(outs[d], truth)]
        ans = [ok for ok, o in zip(right, outs[d]) if not o["abstain"]]
        print("  %-9s all %6.2f%%   answered %6.2f%% (%d of %d)   abstained %5.1f%%" % (
            d, 100 * np.mean(right), 100 * np.mean(ans) if ans else 0.0, len(ans), len(right), 100 * abst[d]))

    # the judges' score, same metric code as selfcheck.py (latency from the p95s measured here)
    import selfcheck as sc
    from decide.scoring import score
    mean_p95 = float(np.mean([pct(times[d], 95) for d in DECISIONS]))
    lat = 1.0 if mean_p95 <= sc.LAT_FULL else max(0.0, (sc.LAT_GATE - mean_p95) / (sc.LAT_GATE - sc.LAT_FULL))
    # the scorer reads only labels; position k holds each decision's k-th call in call order
    aligned = [{"labels": {d: recs[order[d][k]]["labels"][d] for d in DECISIONS}} for k in range(len(recs))]
    total, parts, _ = score(aligned, outs, latency_score=lat)
    print("\n" + c("SCORE (judges' formula)", "1"))
    print("  quality %.3f x40   calibration %.3f x25   latency %.3f x20   abstention %.3f x15" % (
        parts["quality"], parts["calibration"], parts["latency"], parts["abstention"]))
    print("  " + c("TOTAL %.2f / 100" % total, "1;32") + "   (in-process latency; over HTTP add ~3 ms per call)")


if __name__ == "__main__":
    main()
