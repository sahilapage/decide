#!/usr/bin/env python3
"""
Three-minute live demo against the running service.

  python demo.py [--url http://localhost:8000/v1/decide]

1. A normal ticket: all three decisions from the one model, with latency.
2. A case where it correctly refuses to answer (vague ticket -> abstain).
3. Grounding catching a single wrong fact.
4. Approval near a tier's amount cap, where the 3:1 cost rule holds back.
5. A hand-typed, off-template ticket (what a judge might type) still works.
"""

import argparse
import http.client
import json
import time
import urllib.parse

CASES = [
    ("1. Normal ticket - one model, three typed answers", {
        "subject": "Charged twice this month",
        "body": "Hi, I was billed twice for my Pro plan on the 3rd. Could you check and reverse one of them? Thanks.",
        "customer_tier": "pro", "amount": 29.0,
        "draft_reply": "Sorry about that. Duplicate charges are refunded within 5 business days to the original payment method.",
        "kb_article": "Duplicate charges are refunded to the original payment method within five business days of confirmation.",
    }, ["route", "approve", "grounded"]),
    ("2. Correct refusal - the ticket says nothing about which queue", {
        "subject": "Problem",
        "body": "Hi, Can someone help with my account please. Let me know.",
        "customer_tier": "free", "amount": 0.0, "draft_reply": "", "kb_article": "",
    }, ["route"]),
    ("3. Grounding - the draft changes one fact (who the policy applies to)", {
        "subject": "Cancel", "body": "How do I cancel?", "customer_tier": "pro", "amount": 0.0,
        "draft_reply": "Cancelling is available to monthly subscribers with 14 days notice. Your account stays live until the last day of the month and there is no cancellation fee.",
        "kb_article": "annual subscribers may cancel any time up to fourteen days ahead of renewal. Service runs until the last day of the month with no cancellation fee.",
    }, ["grounded"]),
    ("3b. Same draft, article now agrees -> grounded", {
        "subject": "Cancel", "body": "How do I cancel?", "customer_tier": "pro", "amount": 0.0,
        "draft_reply": "Cancelling is available to monthly subscribers with 14 days notice. Your account stays live until the last day of the month and there is no cancellation fee.",
        "kb_article": "monthly subscribers may cancel any time up to two weeks ahead of renewal. Service runs until the last day of the month with no cancellation fee.",
    }, ["grounded"]),
    ("4. Approval: same refund, three amounts on the Free tier", None, ["approve"]),
    ("5. Off-template, hand-typed: short article, courtesy opener, order number in the draft", {
        "subject": "refund status", "body": "Where is my refund for order 48213?", "customer_tier": "pro",
        "amount": 45.5,
        "draft_reply": "Thanks for getting in touch. I've checked order #48213: refunds are issued within 5 business days.",
        "kb_article": "Refunds are issued within five business days of approval.",
    }, ["grounded"]),
    ("5b. Same, but the draft promises 10 business days", {
        "subject": "refund status", "body": "Where is my refund for order 48213?", "customer_tier": "pro",
        "amount": 45.5,
        "draft_reply": "Thanks for getting in touch. I've checked order #48213: refunds are issued within 10 business days.",
        "kb_article": "Refunds are issued within five business days of approval.",
    }, ["grounded"]),
]


def call(conn, path, decision, state, rid):
    body = json.dumps({"request_id": rid, "decision": decision, "state": state})
    t0 = time.perf_counter()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    r = json.loads(conn.getresponse().read())
    r["_rtt_ms"] = (time.perf_counter() - t0) * 1000
    return r


def show(r):
    if r["type"] == "choice":
        probs = "  ".join("%s %.2f" % (k, v) for k, v in sorted(r["probs"].items(), key=lambda kv: -kv[1]))
        print("    %-8s -> %-9s conf %.2f  abstain=%-5s  [%s]  %.1f ms"
              % (r["decision"], r["value"], r["confidence"], r["abstain"], probs, r["_rtt_ms"]))
    else:
        print("    %-8s -> %-9s p_true %.3f  abstain=%-5s  %.1f ms"
              % (r["decision"], r["value"], r["p_true"], r["abstain"], r["_rtt_ms"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/decide")
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    for i, (title, state, decisions) in enumerate(CASES):
        print("\n" + title)
        if state is None:
            for amt in (8.0, 22.0, 60.0):
                s = {"subject": "Refund request", "body": "Please refund my last charge, I did not use it.",
                     "customer_tier": "free", "amount": amt, "draft_reply": "", "kb_article": ""}
                print("   amount $%.2f" % amt, end="")
                show(call(conn, u.path, "approve", s, "demo-%d-%d" % (i, amt)))
            continue
        for d in decisions:
            show(call(conn, u.path, d, state, "demo-%d-%s" % (i, d)))
    print()


if __name__ == "__main__":
    main()
