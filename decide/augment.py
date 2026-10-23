"""
Label-preserving training augmentation, so the encoder stops relying on the
training templates' surface form (exact spelling, KB sentence order, KB length).

The stress test (tools/stress_test.py) showed the template-trained model calls
correct drafts "unsupported" with p ~ 0 when the KB article is reordered,
gets one unrelated sentence, or has typos. Each augmented copy of a training
ticket applies 1-3 of the families below; labels are unchanged.

Families used in training: typo, case, kb_shuffle, kb_distractor.
Deliberately NOT used, so the stress test can measure generalisation to unseen
shifts: number words, punctuation removal, synonyms, body/draft reordering,
amount formatting.
"""

import json
import random
import re

from decide.normalize import normalize
from decide.textfmt import fact_supported, number_facts

_SENT = re.compile(r"(?<=[.!?])\s+")
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
FIELDS = ("subject", "body", "draft_reply", "kb_article")


def _sents(t):
    return [s for s in _SENT.split((t or "").strip()) if s]


def typo(text, rng, rate):
    out = []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(len(w))
            op = rng.randrange(4)
            if op == 0:                                   # delete
                w = w[:i] + w[i + 1:]
            elif op == 1:                                 # insert
                w = w[:i] + rng.choice(_LETTERS) + w[i:]
            elif op == 2:                                 # substitute
                w = w[:i] + rng.choice(_LETTERS) + w[i + 1:]
            elif i < len(w) - 1:                          # swap neighbours
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def case(text, rng):
    k = rng.randrange(3)
    if k == 0:
        return text.lower()
    if k == 1:
        return text.upper() if len(text) < 40 else text
    return " ".join(w.capitalize() if rng.random() < 0.3 else w for w in text.split(" "))


def augment_row(row, rng, kb_pool):
    r = json.loads(json.dumps(row))
    s = r["state"]
    fams = rng.sample(["typo", "case", "kb_shuffle", "kb_distractor"], rng.randint(1, 3))
    for f in fams:
        if f == "typo":
            rate = rng.uniform(0.02, 0.08)
            for k in rng.sample(FIELDS, rng.randint(1, 4)):
                s[k] = typo(str(s.get(k) or ""), rng, rate)
        elif f == "case":
            for k in rng.sample(FIELDS, rng.randint(1, 4)):
                s[k] = case(str(s.get(k) or ""), rng)
        elif f == "kb_shuffle":
            ss = _sents(s.get("kb_article"))
            rng.shuffle(ss)
            s["kb_article"] = " ".join(ss)
        elif f == "kb_distractor":
            d_facts = number_facts(normalize(str(s.get("draft_reply") or "")))
            for _ in range(5):
                cand = rng.choice(_sents(rng.choice(kb_pool)) or [""])
                # never let the extra sentence supply a number the draft states
                if cand and not any(fact_supported(f_, number_facts(normalize(cand))) for f_ in d_facts):
                    ss = _sents(s.get("kb_article"))
                    ss.insert(rng.randint(0, len(ss)), cand)
                    s["kb_article"] = " ".join(ss)
                    break
    return r


def augment(rows, copies, seed=0):
    """Returns `copies` augmented copies of every row (originals not included)."""
    rng = random.Random(seed)
    pool = [str(r["state"].get("kb_article") or "") for r in rows]
    return [augment_row(r, rng, pool) for _ in range(copies) for r in rows]
