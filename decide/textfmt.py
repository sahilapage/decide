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

import numpy as np

from decide.normalize import normalize

DECISIONS = ("route", "approve", "grounded")
TIERS = ["free", "pro", "enterprise"]
AMOUNT_EDGES = [0.0, 0.01, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 110, 120, 150,
                200, 300, 350, 375, 400, 500, 1e12]
TAB_DIM = 3 + 3 * (len(AMOUNT_EDGES) - 1) + 1

PROMPTS = {
    "route": "Which support queue should handle this ticket: billing, technical, account, shipping or general?",
    "approve": "Can the requested action be auto-approved without human review?",
    "grounded": "Is every claim in this draft reply supported by the knowledge-base article?",
}


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
    v[-1] = np.log1p(amt) / 7.0
    return v


def format_input(decision: str, state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    subj = normalize(str(state.get("subject") or ""))
    body = normalize(str(state.get("body") or ""))
    if decision == "route":
        return "%s Subject: %s. Message: %s" % (PROMPTS["route"], subj, body), None
    if decision == "approve":
        tier = str(state.get("customer_tier") or "free").lower()
        try:
            amt = float(state.get("amount") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        return ("%s Tier: %s. Amount: $%.2f. Subject: %s. Message: %s"
                % (PROMPTS["approve"], tier, amt, subj, body)), None
    if decision == "grounded":
        draft = normalize(str(state.get("draft_reply") or ""))
        kb = normalize(str(state.get("kb_article") or ""))
        return "%s Draft: %s" % (PROMPTS["grounded"], draft), "Article: %s" % kb
    raise ValueError(decision)
