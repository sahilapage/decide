"""
Two guardrail rules applied after the model. Both are derived from the
training data only, and neither changes any answer on train or dev.

* red_flag(state): the ticket contains one of the four disqualifying phrase
  types (a repeat request, a brand-new account, a request for an exception, a
  chargeback / bank dispute). Every one of the ~450 training tickets with such
  a phrase was refused, but the network learned the exact template wording, so
  a reworded one ("I raised this before", "I'll dispute it with my card
  company") could be auto-approved at p ~ 0.85. Approve then holds.

* kb_novelty(state): the share of the help article's content words that never
  occur in any training article (decide/kb_idf.json, built from the training
  KB sentences). Every train and dev article scores 0.0. On an article about a
  topic the model has never seen, the grounded head is close to guessing but
  still reports p ~ 0 or 1, so its confidence is pulled towards 0.5 (the
  answer itself is unchanged).
"""

import json
import os
import re
from typing import Any, Dict

from decide.normalize import normalize
from decide.textfmt import PHRASES, kb_terms

_RED_FLAGS = [re.compile(p) for p in PHRASES[:4]]     # repeat, new account, exception, chargeback / bank

NOVEL_KB = 0.15        # article counts as unfamiliar above this share of unseen words
NOVEL_SHRINK = 0.4     # p -> 0.5 + 0.4 * (p - 0.5) on unfamiliar articles
_VOCAB = set()


def red_flag(state: Dict[str, Any]) -> bool:
    text = ("%s %s" % (state.get("subject") or "", state.get("body") or "")).lower()
    return any(r.search(text) for r in _RED_FLAGS)


def kb_novelty(state: Dict[str, Any]) -> float:
    if not _VOCAB:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_idf.json")
        _VOCAB.update(json.load(open(path))["idf"])
    words = {t for t in kb_terms(normalize(str(state.get("kb_article") or ""))) if not t.isdigit()}
    return sum(1 for t in words if t not in _VOCAB) / max(1, len(words))
