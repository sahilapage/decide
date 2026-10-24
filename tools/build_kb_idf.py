"""
Build decide/kb_idf.json: how rare each word stem is across the training KB
sentences. decide/textfmt.py focus_kb uses it to tell a sentence that shares the
draft's *specific* words (seats, administrators, renewal) from one that only
shares words every article uses (days, within, business).

  python tools/build_kb_idf.py            # reads data/train.jsonl
"""

import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.textfmt import _KB_SENT, kb_terms  # noqa: E402


def main():
    rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8") if l.strip()]
    sents = [s for r in rows for s in _KB_SENT.split(str(r["state"]["kb_article"] or "").strip()) if s]
    df = Counter()
    for s in sents:
        df.update({t for t in kb_terms(s) if not t.isdigit()})
    out = {"n_sentences": len(sents),
           "idf": {t: round(math.log(len(sents) / c), 3) for t, c in sorted(df.items())}}
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decide", "kb_idf.json")
    json.dump(out, open(path, "w"), indent=0, sort_keys=True)
    print("wrote %s: %d stems from %d KB sentences" % (path, len(df), len(sents)))


if __name__ == "__main__":
    main()
