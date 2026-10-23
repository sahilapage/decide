"""
Decide: one shared model, three typed decisions.

DecisionModel loads one backbone at startup and answers route / approve /
grounded with it. Two interchangeable backbones exist:

  * "student"  - the fine-tuned transformer (ONNX, CPU), used when
                 artifacts/student/ exists. This is the submission.
  * "baseline" - the shared sparse n-gram backbone with three linear heads,
                 kept as the gate-1 fallback.

Either way, raw logits go through decide.policy.Policy, which applies the
per-decision calibration, the 3:1-cost approve threshold and the abstain rule.
"""

import os
import pickle
from typing import Any, Dict, List, Tuple

from decide.policy import Policy

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        student_dir = os.path.join(artifacts_dir, "student")
        if os.path.exists(os.path.join(student_dir, "model.onnx")) and os.environ.get("DECIDE_BACKEND") != "baseline":
            from decide.student_runtime import StudentRuntime
            self.backend = "student"
            self.net = StudentRuntime(student_dir)
            self.policy = Policy.load(os.path.join(student_dir, "policy.json"))
        else:
            self.backend = "baseline"
            with open(os.path.join(artifacts_dir, "baseline.pkl"), "rb") as fh:
                self.net = pickle.load(fh)
            self.policy = Policy.load(os.path.join(artifacts_dir, "baseline_policy.json"))
        # warm every code path once so the first real request is not slow
        warm = {"subject": "Refund", "body": "Please refund my last charge.", "customer_tier": "pro",
                "amount": 12.5, "draft_reply": "Refunds take 5 days.", "kb_article": "Refunds take 5 days."}
        for _ in range(3):
            self.route(warm), self.approve(warm), self.grounded(warm)

    # -- raw logits from the shared backbone ---------------------------------
    def _logit(self, decision: str, state: Dict[str, Any]):
        if self.backend == "student":
            return self.net.logits(decision, state)
        if decision == "route":
            return self.net.h_route.decision_function(self.net.bb.route_x([state]))[0]
        if decision == "approve":
            return float(self.net.h_appr.decision_function(self.net.bb.approve_x([state]))[0])
        return float(self.net.h_gr.decision_function(self.net.bb.grounded_x([state]))[0])

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        value, probs, conf, abstain = self.policy.route(self._logit("route", state))
        return value, probs, conf, bool(abstain)

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        value, p, abstain = self.policy.approve(self._logit("approve", state))
        return bool(value), p, bool(abstain)

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        value, p, abstain = self.policy.grounded(self._logit("grounded", state))
        return bool(value), p, bool(abstain)
