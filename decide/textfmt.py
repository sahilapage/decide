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

from decide.normalize import damerau1, normalize

DECISIONS = ("route", "approve", "grounded")
TIERS = ["free", "pro", "enterprise"]
AMOUNT_EDGES = [0.0, 0.01, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 110, 120, 150,
                200, 300, 350, 375, 400, 500, 1e12]
APPROVE_DIM = 3 + 3 * (len(AMOUNT_EDGES) - 1) + 1
GROUNDED_DIM = 4
TAB_DIM = APPROVE_DIM + GROUNDED_DIM          # layout v1 (71); v2 appends APPROVE_EXTRA_DIM

# Layout v2 adds approve features so the shared network itself models the
# amount fall-off around each tier's cap and the disqualifying phrases (this
# replaced a separate boosted stage that sat outside the ONNX graph).
# Caps: where approval of requests with no disqualifying phrase falls through
# ~50% in train (0.9-1.0x cap 58%, 1.0-1.1x cap 30%; docs/ceiling.md).
CAPS = {"free": 25.0, "pro": 110.0, "enterprise": 385.0}
CAP_KNOTS = (0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)
# Mined from train labels: phrases that occur only in refused requests. The
# alternatives broaden each one to paraphrases; on train+dev they match exactly
# the same rows as the literal phrases (so training is unchanged), but a
# hand-typed "chargeback" or "make an exception" now reads the same way.
PHRASES = [
    r"raised (something|this|it) (similar|before)|asked (about )?(this|it) before|(contacted|emailed|written to) you "
    r"(about (this|it) )?before|not the first time|again for the same",
    r"signed up recently|(just|only|recently) (signed up|joined|registered|created (my|an|the) account)|"
    r"new (customer|account)|account is (brand )?new",
    r"outside (your|the) (usual )?policy|(appreciate|make|grant|allow) an exception|bend the rules|one-off exception|"
    r"as a one-off",
    r"with my bank|dispute (a|the|this) charge|charge ?back|card (company|issuer)|reverse (it|the charge) (with|through) my bank",
    r"question is",
]
_PHRASE_RE = [re.compile(p) for p in PHRASES]
APPROVE_EXTRA_DIM = 3 * (1 + len(CAP_KNOTS)) + 1 + len(PHRASES)
_TAB = {"version": 1}


def set_tab_version(v: int) -> None:
    """Training and serving must use the same layout; it is stored in args.json / meta.json."""
    _TAB["version"] = int(v)


def tab_dim() -> int:
    return TAB_DIM + (APPROVE_EXTRA_DIM if _TAB["version"] >= 2 else 0)
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


_NOT_QUALIFIER = {"the", "a", "an", "our", "your", "their", "its", "of", "for", "to", "and", "or", "is", "are", "be",
                  "on", "in", "at", "by", "with", "every", "each", "any", "after", "before", "from", "per", "this",
                  "that", "these", "those", "all", "than", "until", "within", "into", "over", "under", "about", "via",
                  "no", "not"}
_WEEKEND = {"weekend", "weekends", "saturday", "saturdays", "sunday", "sundays"}
_WEEKDAY = {"weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday", "mondays", "fridays"}


def scope_broadened_v2(draft: str, kb: str) -> bool:
    """Draft says "all N" where the article only ever qualifies N ("premium N",
    "paid N") and never says "all N". Generalises scope_broadened: on train+dev it
    fires on 0 of 4,822 supported drafts and on 173 unsupported ones (old: 8)."""
    dw = _WORD.findall(draft.lower())
    kw = _WORD.findall(kb.lower())
    kt = " ".join(kw)
    for i in range(len(dw) - 1):
        if dw[i] == "all":
            n = dw[i + 1]
            if n in _NOT_QUALIFIER or ("all " + n) in kt:
                continue
            for j in range(1, len(kw)):
                if kw[j] == n and kw[j - 1] not in _NOT_QUALIFIER:
                    return True
    return False


def days_conflict(draft: str, kb: str) -> bool:
    """Draft promises weekend days the article's opening days exclude, or the reverse."""
    dw, kw = set(_WORD.findall(draft.lower())), set(_WORD.findall(kb.lower()))
    k_days = kw & (_WEEKEND | _WEEKDAY)
    if not k_days:
        return False
    return bool((dw & _WEEKEND) and not (kw & _WEEKEND)) or bool((dw & _WEEKDAY) and not (kw & _WEEKDAY))


def number_facts(text: str) -> set:
    return {(re.sub(r"\s+", "", m.group(1)), m.group(2).lower()) for m in _FACT.finditer(text)}


def _unit_match(a: str, b: str) -> bool:
    """Same unit word, allowing plural/singular and one typo in words of 4+ letters."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 4 and damerau1(a, b):
        return True
    a, b = a.rstrip("s"), b.rstrip("s")
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and damerau1(a, b)


_TICKET_NUM = re.compile(r"\d+(?:\.\d+)?")
_ID_NUM = re.compile(r"#\s*(\d+)")


def ticket_numbers(state: Dict[str, Any]) -> set:
    """Numbers the customer's own ticket contains (subject, body, amount), plus '#123' style IDs in the draft."""
    # Only identifier-like numbers (4+ digits) and money with cents: small
    # integers ("two weeks", "the 14th") collide with real policy numbers.
    nums = set()
    for k in ("subject", "body"):
        nums.update(n for n in _TICKET_NUM.findall(str(state.get(k) or "")) if len(n) >= 4 or "." in n)
    try:
        amt = float(state.get("amount") or 0.0)
        if amt > 0:
            nums.add("%.2f" % amt)
    except (TypeError, ValueError):
        pass
    nums.update(_ID_NUM.findall(str(state.get("draft_reply") or "")))
    return nums


def fact_supported(fact, kb_facts) -> bool:
    # Same number, same unit (one typo allowed). Matching the same duration across
    # units ("1 day" vs "24 hours") was tried and rejected: the check is not
    # claim-aware, so on train+dev it un-flagged 13 unsupported drafts whose
    # changed number happened to equal another claim in another unit.
    num, unit = fact
    return any(n == num and _unit_match(unit, u) for n, u in kb_facts)

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
    v = np.zeros(tab_dim(), dtype=np.float32)
    v[ti] = 1.0
    v[3 + ti * nb + b] = 1.0
    v[APPROVE_DIM - 1] = np.log1p(amt) / 7.0
    # grounded: does the draft state a number+unit (or range) the article never
    # states? After normalisation no supported draft in train or dev does.
    dn = normalize(str(state.get("draft_reply") or ""))
    kn = normalize(str(state.get("kb_article") or ""))
    d = number_facts(dn)
    k = number_facts(kn)
    # A number the reply copies from the ticket itself (order "#48213", the
    # charged amount) is not a policy claim, so the article need not state it.
    tnums = ticket_numbers(state)
    extra = {f for f in d if not fact_supported(f, k) and f[0] not in tnums}
    v[APPROVE_DIM] = 1.0 if extra else 0.0
    v[APPROVE_DIM + 1] = min(len(extra), 3) / 3.0
    v[APPROVE_DIM + 2] = (1 - len(extra) / len(d)) if d else 1.0
    if _TAB["version"] >= 3:
        # one "draft claims a wider / different scope than the article" flag
        v[APPROVE_DIM + 3] = 1.0 if (scope_broadened(dn, kn) or scope_broadened_v2(dn, kn)
                                     or days_conflict(dn, kn)) else 0.0
    else:
        v[APPROVE_DIM + 3] = 1.0 if scope_broadened(dn, kn) else 0.0
    if _TAB["version"] >= 2:
        o = TAB_DIM
        r = amt / CAPS[TIERS[ti]] if amt > 0 else 0.0
        per = 1 + len(CAP_KNOTS)
        b = o + ti * per
        v[b] = min(r, 3.0) / 3.0
        for j, k in enumerate(CAP_KNOTS):
            v[b + 1 + j] = min(max(r - k, 0.0), 2.0)
        o += 3 * per
        v[o] = 1.0 if amt == 0 else 0.0
        text = (str(state.get("subject") or "") + " " + str(state.get("body") or "")).lower()
        for j, rx in enumerate(_PHRASE_RE):
            v[o + 1 + j] = 1.0 if rx.search(text) else 0.0
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
