"""
How each typed question is presented to the one shared encoder.

Every decision is a (prompt, state fields) pair. The prompt names the decision
first, so the same weights read "which queue", "can this be auto-approved"
and "does the article support the draft" as different questions over the same
state. Adding a fourth decision means adding one more prompt here and one small
head, not a new model.

  route     single text: question + subject + body
  approve   single text: question + tier + amount + subject + body
            (+ a small tabular vector: tier one-hot, tier x amount bucket)
  grounded  text pair:   (question + draft reply, kb article) so the encoder
            attends across both and can compare them fact by fact
"""

from typing import Any, Dict, Optional, Tuple

import re

import numpy as np

from decide.normalize import normalize

DECISIONS = ("route", "approve", "grounded")
TIERS = ["free", "pro", "enterprise"]
AMOUNT_EDGES = [0.0, 0.01, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 110, 120, 150,
                200, 300, 350, 375, 400, 500, 1e12]
APPROVE_DIM = 3 + 3 * (len(AMOUNT_EDGES) - 1) + 1
GROUNDED_DIM = 4
TAB_DIM = APPROVE_DIM + GROUNDED_DIM
# a number (or range "3-5") together with the word that follows it, e.g.
# ("3-5", "business"), ("300", "requests"), ("5", "minutes")
_FACT = re.compile(r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*([a-z%$]*)", re.IGNORECASE)


_WORD = re.compile(r"[a-z]+")


def scope_broadened(draft: str, kb: str) -> bool:
    """
    True when the draft says "all N" but the article only says "all <q> N"
    for the same noun - e.g. article "all paid plans", draft "all plans".
    Dropping the qualifier widens who the policy covers. (Narrowing the other
    way round is allowed by the labels, so only this direction is flagged.)
    """
    dw = _WORD.findall(draft.lower())
    kw = _WORD.findall(kb.lower())
    kt = " ".join(kw)
    for i in range(len(dw) - 1):
        if dw[i] == "all" and ("all " + dw[i + 1]) not in kt:
            for j in range(len(kw) - 2):
                if kw[j] == "all" and kw[j + 2] == dw[i + 1]:
                    return True
    return False


def number_facts(text: str) -> set:
    return {(re.sub(r"\s+", "", m.group(1)), m.group(2).lower()) for m in _FACT.finditer(text)}

PROMPT_SETS = {
    # full natural-language questions
    "long": {
        "route": "Which support queue should handle this ticket: billing, technical, account, shipping or general?",
        "approve": "Can the requested action be auto-approved without human review?",
        "grounded": "Is every claim in this draft reply supported by the knowledge-base article?",
    },
    # compact typed-question tags: same role, ~15 fewer tokens per call
    "short": {"route": "route:", "approve": "approve:", "grounded": "grounded:"},
}
PROMPTS = dict(PROMPT_SETS["long"])
_STYLE = {"name": "long"}


def set_prompt_style(name: str) -> None:
    """Training and serving must use the same style; it is stored in meta.json."""
    PROMPTS.clear()
    PROMPTS.update(PROMPT_SETS[name])
    _STYLE["name"] = name


def prompt_style() -> str:
    return _STYLE["name"]


def tabular(state: Dict[str, Any]) -> np.ndarray:
    tier = str(state.get("customer_tier") or "free").lower()
    ti = TIERS.index(tier) if tier in TIERS else 0
    try:
        amt = max(0.0, float(state.get("amount") or 0.0))
    except (TypeError, ValueError):
        amt = 0.0
    nb = len(AMOUNT_EDGES) - 1
    b = int(np.searchsorted(AMOUNT_EDGES, amt, side="right") - 1)
    b = min(max(b, 0), nb - 1)
    v = np.zeros(TAB_DIM, dtype=np.float32)
    v[ti] = 1.0
    v[3 + ti * nb + b] = 1.0
    v[APPROVE_DIM - 1] = np.log1p(amt) / 7.0
    # grounded: does the draft state a number+unit (or range) the article never
    # states? After normalisation no supported draft in train or dev does.
    dn = normalize(str(state.get("draft_reply") or ""))
    kn = normalize(str(state.get("kb_article") or ""))
    d = number_facts(dn)
    k = number_facts(kn)
    extra = d - k
    v[APPROVE_DIM] = 1.0 if extra else 0.0
    v[APPROVE_DIM + 1] = min(len(extra), 3) / 3.0
    v[APPROVE_DIM + 2] = (len(d & k) / len(d)) if d else 1.0
    v[APPROVE_DIM + 3] = 1.0 if scope_broadened(dn, kn) else 0.0
    return v


def format_input(decision: str, state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    subj = normalize(str(state.get("subject") or ""))
    body = normalize(str(state.get("body") or ""))
    if decision == "route":
        if _STYLE["name"] == "short":
            return "%s %s | %s" % (PROMPTS["route"], subj, body), None
        return "%s Subject: %s. Message: %s" % (PROMPTS["route"], subj, body), None
    if decision == "approve":
        tier = str(state.get("customer_tier") or "free").lower()
        try:
            amt = float(state.get("amount") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if _STYLE["name"] == "short":
            return "%s %s $%.2f | %s | %s" % (PROMPTS["approve"], tier, amt, subj, body), None
        return ("%s Tier: %s. Amount: $%.2f. Subject: %s. Message: %s"
                % (PROMPTS["approve"], tier, amt, subj, body)), None
    if decision == "grounded":
        draft = normalize(str(state.get("draft_reply") or ""))
        kb = normalize(str(state.get("kb_article") or ""))
        if _STYLE["name"] == "short":
            return "%s %s" % (PROMPTS["grounded"], draft), kb
        return "%s Draft: %s" % (PROMPTS["grounded"], draft), "Article: %s" % kb
    raise ValueError(decision)
