"""
Shifted-dev stress test: how much does the SHIPPED model (and its shipped
policy, no refit) lose when dev tickets are rewritten in ways that keep the
label but change the surface form? A proxy for "judges use another dataset".

  python tools/stress_test.py                 # all variants
  python tools/stress_test.py --only typos,kb_shuffle --n 600

Label-preserving variants (should score ~ like clean dev):
  typos          adjacent-letter swaps in ~6% of words (not numbers) in subject/body/draft/kb
  lower_nopunct  lowercase, punctuation stripped (keeps $ . digits)
  synonyms       common support-vocabulary swaps (refund->reimbursement, ...)
  body_shuffle   body sentences in random order
  amount_format  "$136.12" -> "136.12 USD" in the body
  kb_shuffle     KB article sentences in random order
  draft_shuffle  draft sentences in random order
  kb_distractor  one sentence from an unrelated KB article appended
  num_words      digits <-> number words swapped in draft and kb
  draft_ticket_ref  draft opens by quoting the order number / charged amount (not a policy claim)
  kb_single      one-sentence KB article (closest sentence to the draft); correct drafts cut to one sentence
  all_mixed      a random 3 of the above per ticket
Label-flipping variant (tests sensitivity, labels updated):
  num_flip       for grounded=True drafts, one number shared with the KB is changed -> grounded=False
"""

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.scoring import score  # noqa: E402
from model import DecisionModel, QUEUES  # noqa: E402

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen", "twenty"]
TENS = {30: "thirty", 40: "forty", 45: "forty-five", 60: "sixty", 90: "ninety"}
W2N = {w: i for i, w in enumerate(WORDS)}
W2N.update({v: k for k, v in TENS.items()})

SYN = [("refund", "reimbursement"), ("charged", "billed"), ("charge", "payment"), ("package", "parcel"),
       ("order", "purchase"), ("log in", "sign in"), ("login", "sign-in"), ("password", "passcode"),
       ("broken", "not working"), ("error", "fault"), ("subscription", "plan"), ("invoice", "bill"),
       ("delivered", "received"), ("account", "profile"), ("cancel", "stop"), ("help", "assist"),
       ("please", "kindly"), ("issue", "problem"), ("crash", "freeze"), ("shipping", "delivery")]

SENT = re.compile(r"(?<=[.!?])\s+")


def sents(t):
    return [s for s in SENT.split(t.strip()) if s]


def typos(t, rng, rate=0.06):
    out = []
    for w in t.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def lower_nopunct(t, rng):
    return re.sub(r"[^\w\s$.]", "", t.lower())


def synonyms(t, rng):
    for a, b in SYN:
        t = re.sub(r"\b%s\b" % re.escape(a), b, t, flags=re.I)
    return t


def shuffle(t, rng):
    s = sents(t)
    rng.shuffle(s)
    return " ".join(s)


def amount_format(t, rng):
    return re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 USD", t)


def num_words(t, rng):
    def d2w(m):
        n = int(m.group(0))
        return WORDS[n] if n < len(WORDS) else TENS.get(n, m.group(0))

    def w2d(m):
        return str(W2N[m.group(0).lower()])
    t = re.sub(r"\b\d{1,2}\b(?!\.)", d2w, t) if rng.random() < 0.5 else \
        re.sub(r"(?<![-\w])(%s)\b(?![-\s]+(?:hundred|thousand|and)\b|-)" % "|".join(sorted(W2N, key=len, reverse=True)),
               w2d, t, flags=re.I)
    return t


def make_variant(name, recs, rng):
    out = []
    kbs = [r["state"]["kb_article"] for r in recs]
    for r in recs:
        r = json.loads(json.dumps(r))
        s = r["state"]
        if name == "typos":
            for k in ("subject", "body", "draft_reply", "kb_article"):
                s[k] = typos(s[k], rng)
        elif name == "lower_nopunct":
            for k in ("subject", "body", "draft_reply", "kb_article"):
                s[k] = lower_nopunct(s[k], rng)
        elif name == "synonyms":
            s["subject"], s["body"] = synonyms(s["subject"], rng), synonyms(s["body"], rng)
        elif name == "body_shuffle":
            s["body"] = shuffle(s["body"], rng)
        elif name == "amount_format":
            s["body"] = amount_format(s["body"], rng)
        elif name == "kb_shuffle":
            s["kb_article"] = shuffle(s["kb_article"], rng)
        elif name == "draft_shuffle":
            s["draft_reply"] = shuffle(s["draft_reply"], rng)
        elif name == "kb_distractor":
            other = sents(rng.choice(kbs))
            s["kb_article"] = s["kb_article"].rstrip() + " " + rng.choice(other)
        elif name == "num_words":
            s["draft_reply"], s["kb_article"] = num_words(s["draft_reply"], rng), num_words(s["kb_article"], rng)
        elif name == "all_mixed":
            for v in rng.sample(LABEL_PRESERVING[:-1], 3):
                r = make_variant(v, [r], rng)[0]
                s = r["state"]
        elif name == "draft_ticket_ref":
            ref = "I've noted this on order #%d." % rng.randint(10000, 99999)
            if float(s.get("amount") or 0) > 0:
                ref = "I've looked into the $%.2f charge on order #%d." % (float(s["amount"]), rng.randint(10000, 99999))
            s["draft_reply"] = ref + " " + s["draft_reply"]
        elif name == "kb_single":
            # short article, as a person typing a quick example would write: a
            # random draft sentence (correct drafts only) and only the KB
            # sentences it leans on. Close to decide/augment.py's kb_trim (in-family).
            cw = lambda t: set(re.findall(r"[a-z]{4,}", t.lower()))
            ds, ks = sents(s["draft_reply"]), sents(s["kb_article"])
            if r["labels"]["grounded"] and len(ds) > 1:
                s["draft_reply"] = rng.choice([x for x in ds if len(cw(x)) >= 3] or ds)
            if ks:
                d = cw(s["draft_reply"])
                best = max(ks, key=lambda x: len(d & cw(x)))
                if r["labels"]["grounded"]:
                    # keep every sentence the draft leans on, so the label stays true
                    s["kb_article"] = " ".join(x for x in ks if len(d & cw(x)) >= 2) or best
                else:
                    s["kb_article"] = best
        elif name == "num_flip":
            if r["labels"]["grounded"]:
                kbn = set(re.findall(r"\b\d+\b", s["kb_article"]))
                shared = [m for m in re.finditer(r"\b\d+\b", s["draft_reply"]) if m.group(0) in kbn]
                if shared:
                    m = rng.choice(shared)
                    n = int(m.group(0))
                    new = str(n * 2 if n < 50 else n + 7)
                    s["draft_reply"] = s["draft_reply"][:m.start()] + new + s["draft_reply"][m.end():]
                    r["labels"]["grounded"] = False
        out.append(r)
    return out


LABEL_PRESERVING = ["typos", "lower_nopunct", "synonyms", "body_shuffle", "amount_format",
                    "kb_shuffle", "draft_shuffle", "kb_distractor", "num_words", "draft_ticket_ref", "kb_single", "all_mixed"]


def run(model, recs):
    outs = {"route": [], "approve": [], "grounded": []}
    for r in recs:
        s = r["state"]
        v, probs, _, ab = model.route(s)
        outs["route"].append({"value": v, "probs": probs, "abstain": ab})
        v, p, ab = model.approve(s)
        outs["approve"].append({"value": v, "p_true": p, "abstain": ab})
        v, p, ab = model.grounded(s)
        outs["grounded"].append({"value": v, "p_true": p, "abstain": ab})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev.jsonl")
    ap.add_argument("--n", type=int, default=0, help="subsample (0 = all)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.n:
        recs = random.Random(0).sample(recs, args.n)
    model = DecisionModel(args.artifacts)
    names = ["clean"] + LABEL_PRESERVING + ["num_flip"]
    if args.only:
        names = ["clean"] + args.only.split(",")
    rows = []
    for name in names:
        rng = random.Random(1)
        data = recs if name == "clean" else make_variant(name, recs, rng)
        t0 = time.time()
        total, parts, d = score(data, run(model, data))
        row = dict(variant=name, total=round(total, 2), route_acc=round(d["route_acc"], 4),
                   approve_acc=round(d["appr_acc"], 4), grounded_acc=round(d["gr_acc"], 4),
                   route_ece=round(d["route_ece"], 4), appr_ece=round(d["appr_ece"], 4),
                   gr_ece=round(d["gr_ece"], 4), **{k: round(v, 4) for k, v in parts.items()})
        rows.append(row)
        print("%-14s total %6.2f | route %.3f approve %.3f grounded %.3f | ECE r %.3f a %.3f g %.3f | "
              "abst %.3f  (%.0fs)" % (name, total, d["route_acc"], d["appr_acc"], d["gr_acc"], d["route_ece"],
                                      d["appr_ece"], d["gr_ece"], parts["abstention"], time.time() - t0),
              flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
