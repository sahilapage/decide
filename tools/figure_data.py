"""
Run the shipped model in-process and save the numbers behind the model-driven
figures (calibration, accuracy vs coverage, the approve curve) to
docs/figures/data/model_dev.json. tools/make_figures.py plots them.

  python tools/figure_data.py            # serving venv: numpy, onnxruntime, tokenizers
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import DecisionModel  # noqa: E402

# the refund ticket from demo case 4, swept over amounts on every tier
SWEEP_TICKET = {"subject": "Refund request", "body": "Please refund my last charge, I did not use it.",
                "draft_reply": "", "kb_article": ""}
SWEEP_MAX = {"free": 60.0, "pro": 220.0, "enterprise": 800.0}


def main():
    m = DecisionModel(os.path.join(ROOT, "artifacts"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "dev.jsonl")) if l.strip()]
    out = {"route": [], "approve": [], "grounded": [], "sweep": {}}
    for r in rows:
        s, y = r["state"], r["labels"]
        v, _, conf, ab = m.route(s)
        out["route"].append([round(conf, 5), int(v == y["route"]), int(ab)])
        v, p, ab = m.approve(s)
        out["approve"].append([round(p, 5), int(bool(y["approve"])), int(v == bool(y["approve"])), int(ab)])
        v, p, ab = m.grounded(s)
        out["grounded"].append([round(p, 5), int(bool(y["grounded"])), int(v == bool(y["grounded"])), int(ab)])
    for tier, hi in SWEEP_MAX.items():
        pts = []
        for i in range(121):
            amt = round(hi * i / 120 + 0.37, 2)       # cents, like 99% of real amounts (see model card)
            v, p, ab = m.approve(dict(SWEEP_TICKET, customer_tier=tier, amount=amt))
            pts.append([amt, round(p, 5), int(v), int(ab)])
        out["sweep"][tier] = pts
    path = os.path.join(ROOT, "docs", "figures", "data", "model_dev.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))
    print("wrote", os.path.relpath(path, ROOT), "| %d dev rows" % len(rows))


if __name__ == "__main__":
    main()
