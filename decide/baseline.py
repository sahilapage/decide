"""
Gate-1 baseline: one shared sparse backbone, three linear heads.

Backbone: a single binary n-gram vocabulary (1-3 grams over normalised text)
fitted once on every text field. All three heads read features built from that
one vocabulary:

  route     B(subject + body)
  approve   B(subject + body) + tier one-hot + tier x amount-bucket one-hot
  grounded  B(draft) AND NOT B(kb)    (draft phrases the article never says)
            B(draft) AND B(kb)        (phrases they share)
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from decide.normalize import normalize

QUEUES = ["billing", "technical", "account", "shipping", "general"]
TIERS = ["free", "pro", "enterprise"]
AMOUNT_EDGES = [0.0, 0.01, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 110, 120, 150, 200, 300, 350, 375, 400, 500, 1e12]


def ticket_text(s):
    return normalize(s.get("subject", "") + " . " + s.get("body", ""))


def tabular(s):
    tier = s.get("customer_tier", "free")
    ti = TIERS.index(tier) if tier in TIERS else 0
    amt = float(s.get("amount") or 0.0)
    b = int(np.searchsorted(AMOUNT_EDGES, amt, side="right") - 1)
    b = min(max(b, 0), len(AMOUNT_EDGES) - 2)
    nb = len(AMOUNT_EDGES) - 1
    v = np.zeros(3 + 3 * nb + 1, dtype=np.float32)
    v[ti] = 1
    v[3 + ti * nb + b] = 1
    v[-1] = np.log1p(amt) / 7.0
    return v


class SparseBackbone:
    def __init__(self):
        self.vec = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True,
                                   token_pattern=r"(?u)\b\w+\b", dtype=np.float32)

    def fit(self, states):
        corpus = []
        for s in states:
            corpus += [ticket_text(s), normalize(s.get("draft_reply", "")), normalize(s.get("kb_article", ""))]
        self.vec.fit(corpus)
        return self

    def route_x(self, states):
        return self.vec.transform([ticket_text(s) for s in states]).tocsr()

    def approve_x(self, states):
        t = self.vec.transform([ticket_text(s) for s in states])
        tab = csr_matrix(np.stack([tabular(s) for s in states]))
        return hstack([t, tab]).tocsr()

    def grounded_x(self, states):
        d = self.vec.transform([normalize(s.get("draft_reply", "")) for s in states])
        k = self.vec.transform([normalize(s.get("kb_article", "")) for s in states])
        shared = d.multiply(k)
        missing = d - shared
        n_missing = csr_matrix(np.asarray(missing.sum(1)) / 10.0)
        return hstack([missing, shared, n_missing]).tocsr()


class BaselineModel:
    def __init__(self, C_route=0.5, C_appr=0.5, C_gr=1.0):
        self.bb = SparseBackbone()
        self.h_route = LogisticRegression(C=C_route, max_iter=4000)
        self.h_appr = LogisticRegression(C=C_appr, max_iter=4000)
        self.h_gr = LogisticRegression(C=C_gr, max_iter=4000)

    def fit(self, rows):
        st = [r["state"] for r in rows]
        self.bb.fit(st)
        self.h_route.fit(self.bb.route_x(st), [QUEUES.index(r["labels"]["route"]) for r in rows])
        self.h_appr.fit(self.bb.approve_x(st), [int(r["labels"]["approve"]) for r in rows])
        self.h_gr.fit(self.bb.grounded_x(st), [int(r["labels"]["grounded"]) for r in rows])
        return self

    def logits(self, states):
        r = self.h_route.decision_function(self.bb.route_x(states))
        a = self.h_appr.decision_function(self.bb.approve_x(states))
        g = self.h_gr.decision_function(self.bb.grounded_x(states))
        return r, a, g
