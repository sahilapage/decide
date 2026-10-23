"""
HTTP layer for the Decide service. You should not need to change this file.

It implements the API contract from the problem statement exactly:

    POST /v1/decide
    { "request_id": ..., "decision": "route"|"approve"|"grounded", "state": {...} }

and dispatches to DecisionModel in model.py. If you change the response shape
here, automated judging will reject your submission - so if something needs to
change, change model.py.

Run it with:

    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from model import QUEUES, DecisionModel

PROB_TOLERANCE = 0.01
MODEL: Optional[DecisionModel] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL
    MODEL = DecisionModel(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    yield


app = FastAPI(title="Decide", docs_url=None, redoc_url=None, lifespan=lifespan)


class State(BaseModel):
    subject: str = ""
    body: str = ""
    customer_tier: str = "free"
    amount: float = 0.0
    draft_reply: str = ""
    kb_article: str = ""

    model_config = {"extra": "allow"}

    # An explicit null in a field is coerced rather than rejected. A 422 here
    # would count as a malformed response and fail the schema gate.
    @field_validator("subject", "body", "customer_tier", "draft_reply",
                     "kb_article", mode="before")
    @classmethod
    def _blank_if_null(cls, v: Any) -> Any:
        return "" if v is None else v

    @field_validator("amount", mode="before")
    @classmethod
    def _zero_if_null(cls, v: Any) -> Any:
        return 0.0 if v is None else v


class DecideRequest(BaseModel):
    request_id: str
    decision: Literal["route", "approve", "grounded"]
    state: State


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "loaded": MODEL is not None}


@app.post("/v1/decide")
def decide(req: DecideRequest) -> JSONResponse:
    t0 = time.perf_counter()
    state = req.state.model_dump()

    if req.decision == "route":
        value, probs, confidence, abstain = MODEL.route(state)
        probs = {q: float(probs.get(q, 0.0)) for q in QUEUES}
        total = sum(probs.values())
        if total > 0 and abs(total - 1.0) > PROB_TOLERANCE:
            # Renormalise rather than emit an invalid response. If this fires
            # your model is returning something that does not sum to 1 - worth
            # fixing at source, because the judges check this too.
            probs = {q: p / total for q, p in probs.items()}
        if value not in probs:
            value = max(probs, key=probs.get)
        # The judges check that confidence equals probs[value].
        body: Dict[str, Any] = {
            "type": "choice",
            "value": value,
            "probs": probs,
            "confidence": float(probs[value]),
            "abstain": bool(abstain),
        }
    else:
        fn = MODEL.approve if req.decision == "approve" else MODEL.grounded
        value, p_true, abstain = fn(state)
        body = {
            "type": "binary",
            "value": bool(value),
            "p_true": float(p_true),
            "abstain": bool(abstain),
        }

    body["request_id"] = req.request_id
    body["decision"] = req.decision
    body["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return JSONResponse(body)
