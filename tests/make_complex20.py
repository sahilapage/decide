"""
20 dev-style test tickets with extra complexity: 20 random dev rows, each
rewritten with 2-3 label-preserving changes at once (typos, synonyms,
reordered sentences, "$" amount formats, number words, extra or reordered KB
sentences, an order number in the draft, an email-thread wrapper, a longer
body, a second less-urgent issue, a longer KB). The rewrites are the ones in
tools/stress_test.py, so the labels stay correct. Same format as dev.jsonl.

  python tests/make_complex20.py          # writes tests/complex20.jsonl
  python test.py tests/complex20.jsonl
"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from stress_test import make_variant  # noqa: E402

REWRITES = ["typos", "synonyms", "body_shuffle", "amount_format", "num_words", "kb_shuffle", "kb_distractor",
            "draft_ticket_ref", "thread", "long_body", "multi_issue", "long_kb"]


def main(seed=11, n=20):
    rng = random.Random(seed)
    dev = [json.loads(l) for l in open(os.path.join(ROOT, "data", "dev.jsonl")) if l.strip()]
    picked = rng.sample(dev, n)
    out = []
    for r in picked:
        cur = r
        for v in rng.sample(REWRITES, rng.randint(2, 3)):
            pool = rng.sample(dev, 300)               # donor sentences for the longer / multi-issue rewrites
            cur = make_variant(v, [cur] + pool, rng)[0]
        out.append({"request_id": cur["request_id"], "state": cur["state"], "labels": cur["labels"]})
    path = os.path.join(ROOT, "tests", "complex20.jsonl")
    with open(path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
