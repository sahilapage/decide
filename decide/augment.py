"""
Label-preserving training augmentation, so the encoder stops relying on the
training templates' surface form (exact spelling, KB sentence order, KB length).

The stress test (tools/stress_test.py) showed the template-trained model calls
correct drafts "unsupported" with p ~ 0 when the KB article is reordered,
gets one unrelated sentence, or has typos. Each augmented copy of a training
ticket applies 1-3 of the families below; labels are unchanged.

Families used in training: typo, case, kb_shuffle, kb_distractor, courtesy
(no-claim opener/closer sentences around the draft), kb_trim (KB sentences the
draft does not use are dropped, so one-sentence articles look familiar).
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
FAMILIES = ["typo", "case", "kb_shuffle", "kb_distractor", "courtesy", "kb_trim"]
_CONTENT = re.compile(r"[a-z]{4,}")


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


# Courtesy / reference sentences that make no policy claim. Real replies wrap
# the policy answer in these; a correct draft must stay "grounded". Wording is
# deliberately different from tools/stress_test.py's draft_ticket_ref variant.
_OPENERS = ["Thanks for getting in touch.", "Thank you for your patience.", "Happy to help with this.",
            "I understand how frustrating this is.", "Apologies for the inconvenience.", "Good question.",
            "Thanks for the details.", "I have checked this for you.", "I've reviewed ticket #{id}.",
            "Hi there, thanks for writing in.", "Sorry for the trouble here."]
_CLOSERS = ["Let me know if there is anything else.", "Have a great day.", "Reply here if you need more help.",
            "Your reference is #{id}.", "Hope that helps!", "We appreciate your patience.",
            "Feel free to reach out again.", "Kind regards, the support team."]


def courtesy(draft, rng):
    def fill(t):
        return t.replace("{id}", str(rng.randint(1000, 999999)))
    if rng.random() < 0.6:
        draft = fill(rng.choice(_OPENERS)) + " " + draft
    if rng.random() < 0.6:
        draft = draft.rstrip() + " " + fill(rng.choice(_CLOSERS))
    return draft


def kb_trim(draft, kb, grounded, rng):
    """Drop KB sentences the draft does not lean on (short articles exist in the wild).
    Correct drafts keep every sentence that shares 2+ content words with them;
    an unsupported draft stays unsupported whatever is removed."""
    ss = _sents(kb)
    if len(ss) < 2:
        return kb
    dw = set(_CONTENT.findall(normalize(draft).lower()))
    keep = []
    for x in ss:
        overlap = len(dw & set(_CONTENT.findall(normalize(x).lower())))
        if overlap >= 2 or (not grounded and rng.random() < 0.5):
            keep.append(x)
    best = max(ss, key=lambda x: len(dw & set(_CONTENT.findall(normalize(x).lower()))))
    if not keep or (not grounded and rng.random() < 0.3):
        # unsupported drafts get short articles too, so length is no shortcut
        keep = [best]
    return " ".join(keep)


def augment_row(row, rng, kb_pool):
    r = json.loads(json.dumps(row))
    s = r["state"]
    fams = rng.sample(FAMILIES, rng.randint(1, 3))
    for f in fams:
        if f == "typo":
            rate = rng.uniform(0.02, 0.08)
            for k in rng.sample(FIELDS, rng.randint(1, 4)):
                s[k] = typo(str(s.get(k) or ""), rng, rate)
        elif f == "case":
            for k in rng.sample(FIELDS, rng.randint(1, 4)):
                s[k] = case(str(s.get(k) or ""), rng)
        elif f == "courtesy":
            s["draft_reply"] = courtesy(str(s.get("draft_reply") or ""), rng)
        elif f == "kb_trim":
            g = bool(r["labels"]["grounded"])
            ds = _sents(s.get("draft_reply"))
            if g and len(ds) > 1 and rng.random() < 0.5:
                # a subset of supported claims is still supported (never done for
                # unsupported drafts: the dropped sentence could be the bad claim)
                s["draft_reply"] = max(ds, key=len)
            s["kb_article"] = kb_trim(str(s.get("draft_reply") or ""), str(s.get("kb_article") or ""), g, rng)
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
