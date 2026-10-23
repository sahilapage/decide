"""
Live-demo page for the three-minute demo. Not part of the graded API.

  uvicorn demo_ui:app --host 0.0.0.0 --port 8000
  open http://localhost:8000/

This imports the real service from app.py unchanged and adds:
  GET  /                 a single static page (no external assets)
  POST /demo/baseline    the same request answered by the gate-1 baseline
                         (sparse n-gram backbone, artifacts/baseline.pkl), for
                         a live side-by-side; needs scikit-learn (training deps)
  GET  /demo/minieval    a live mini-evaluation: n random dev tickets through
                         both models, clean and rewritten (tools/stress_test.py),
                         scored with the judges' metric code. Needs data/dev.jsonl.
POST /v1/decide and GET /health are the same routes the judges hit, so
selfcheck.py still works against this port.
"""

import json
import os
import random
import sys
import time

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app import app

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "demo", "index.html")
_BASE = {}


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(PAGE, media_type="text/html")


def _models():
    """(Decide, baseline). The baseline is loaded once, on first use."""
    from model import DecisionModel
    if "decide" not in _BASE:
        _BASE["decide"] = DecisionModel(os.path.join(ROOT, "artifacts"))
    if "baseline" not in _BASE:
        prev = os.environ.get("DECIDE_BACKEND")
        os.environ["DECIDE_BACKEND"] = "baseline"
        try:
            _BASE["baseline"] = DecisionModel(os.path.join(ROOT, "artifacts"))
        except Exception as e:  # scikit-learn missing on a serving-only install
            _BASE["baseline"] = None
            _BASE["baseline_error"] = "%s: %s" % (type(e).__name__, e)
        finally:
            if prev is None:
                os.environ.pop("DECIDE_BACKEND", None)
            else:
                os.environ["DECIDE_BACKEND"] = prev
    return _BASE["decide"], _BASE["baseline"]


def _answer(m, decision, state):
    t0 = time.perf_counter()
    if decision == "route":
        v, probs, conf, ab = m.route(state)
        out = {"value": v, "probs": probs, "confidence": conf, "abstain": ab}
    elif decision in ("approve", "grounded"):
        v, p, ab = getattr(m, decision)(state)
        out = {"value": v, "p_true": p, "abstain": ab}
    else:
        raise HTTPException(400, "unknown decision")
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


@app.post("/demo/baseline", include_in_schema=False)
def demo_baseline(req: dict):
    _, base = _models()
    if base is None:
        raise HTTPException(503, "baseline unavailable (" + _BASE.get("baseline_error", "") + ")")
    return _answer(base, req.get("decision"), req.get("state") or {})


@app.get("/demo/minieval", include_in_schema=False)
def demo_minieval(n: int = 150, seed: int = 0):
    path = os.path.join(ROOT, "data", "dev.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "data/dev.jsonl not found")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from stress_test import make_variant  # noqa: E402
    from decide.scoring import score  # noqa: E402
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = random.Random(seed).sample(rows, min(max(n, 30), 600))
    decide, base = _models()
    sets = {"clean": rows,
            "rewritten": make_variant("all_mixed", rows, random.Random(seed + 1))}
    result = {"n": len(rows), "models": {}}
    for name, m in (("decide", decide), ("baseline", base)):
        if m is None:
            continue
        result["models"][name] = {}
        for sname, data in sets.items():
            outs = {"route": [], "approve": [], "grounded": []}
            lat = []
            for r in data:
                for d in outs:
                    a = _answer(m, d, r["state"])
                    lat.append(a["latency_ms"])
                    outs[d].append(a)
            total, parts, det = score(data, outs)
            lat.sort()
            result["models"][name][sname] = {
                "total": round(total, 2), "route_acc": round(det["route_acc"], 3),
                "approve_acc": round(det["appr_acc"], 3), "grounded_acc": round(det["gr_acc"], 3),
                "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2)}
    return result
