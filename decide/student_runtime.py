"""
CPU runtime for the exported student: Rust tokenizer + one ONNX Runtime session.

Per request: normalise + format the typed question (~20us), tokenise (~50us),
one forward pass of the shared encoder, read the head that was asked for.
"""

import json
import os
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_FIELD_CHARS = 4000

from decide.textfmt import format_input, set_prompt_style, set_tab_version, tabular


class StudentRuntime:
    def __init__(self, folder: str):
        meta = json.load(open(os.path.join(folder, "meta.json")))
        self.use_tt = meta["use_token_type_ids"]
        set_prompt_style(meta.get("prompt_style", "long"))
        set_tab_version(meta.get("tab_version", 1))
        self.tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=int(meta["max_len"]), strategy="longest_first")
        variant = os.environ.get("DECIDE_ONNX", meta.get("variant", "model.onnx"))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(os.environ.get("DECIDE_THREADS", meta.get("threads", 4)))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # spinning keeps idle threads hot; it hurts when threads > physical cores
        so.add_session_config_entry("session.intra_op.allow_spinning",
                                    os.environ.get("DECIDE_SPIN", "1"))
        path = os.path.join(folder, variant)
        if os.path.exists(path):
            src = path
        else:
            # stored as <100MB chunks so it fits in a plain git repo; rejoin in memory
            parts = sorted(p for p in os.listdir(folder) if p.startswith(variant + ".part"))
            if not parts:
                raise FileNotFoundError(path)
            src = b"".join(open(os.path.join(folder, p), "rb").read() for p in parts)
        self.sess = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        self.out_names = ["route", "approve", "grounded"]

    def encode(self, decision: str, state: Dict[str, Any]):
        a, b = format_input(decision, state)
        enc = self.tok.encode(a, b) if b is not None else self.tok.encode(a)
        ids = np.asarray([enc.ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids),
                "tab": tabular(state)[None, :]}
        if self.use_tt:
            feed["token_type_ids"] = np.asarray([enc.type_ids], dtype=np.int64)
        return feed

    def logits(self, decision: str, state: Dict[str, Any]):
        # Clip very long fields before tokenising: pair truncation is quadratic in
        # length, so a pasted full article could otherwise cost >100 ms. 4,000
        # characters is ~8x the longest field in train/dev (482).
        state = {k: (v[:MAX_FIELD_CHARS] if isinstance(v, str) else v) for k, v in (state or {}).items()}
        r, a, g = self.sess.run(self.out_names, self.encode(decision, state))
        if decision == "route":
            return r[0].astype(np.float64)
        if decision == "approve":
            return float(a[0])
        return float(g[0])
