"""
Compare ONNX variants of one exported student on the full dev set:
fp32 vs several int8 dynamic-quantisation recipes. Runs every dev call through
the real CPU runtime, then scores with a cross-fitted policy (fit_policy_dev).

  python tools/quant_eval.py exports/e32s
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.student_runtime import StudentRuntime  # noqa: E402


def make_variants(folder):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src = os.path.join(folder, "model.onnx")
    recipes = {
        "q8_matmul_pc.onnx": dict(per_channel=True, op_types_to_quantize=["MatMul"]),
        "q8_matmul_pt.onnx": dict(per_channel=False, op_types_to_quantize=["MatMul"]),
        "q8_matmul_pc_rr.onnx": dict(per_channel=True, reduce_range=True, op_types_to_quantize=["MatMul"]),
    }
    for name, kw in recipes.items():
        out = os.path.join(folder, name)
        if not os.path.exists(out):
            quantize_dynamic(src, out, weight_type=QuantType.QInt8, **kw)
    return ["model.onnx", "model.int8.onnx"] + list(recipes)


def main():
    folder = sys.argv[1]
    rows = [json.loads(l) for l in open("data/dev.jsonl")]
    for var in make_variants(folder):
        os.environ["DECIDE_ONNX"] = var
        os.environ["DECIDE_THREADS"] = "4"
        rt = StudentRuntime(folder)
        R, A, G, lat = [], [], [], []
        for r in rows:
            for d in ("route", "approve", "grounded"):
                t0 = time.perf_counter()
                z = rt.logits(d, r["state"])
                lat.append((time.perf_counter() - t0) * 1000)
                (R if d == "route" else A if d == "approve" else G).append(z)
        out = os.path.join(folder, "dev_onnx_%s.npz" % var.replace(".onnx", ""))
        np.savez(out, route=np.asarray(R), approve=np.asarray(A), grounded=np.asarray(G))
        res = subprocess.run([sys.executable, "training/fit_policy_dev.py", out, "--out", out + ".policy.json"],
                             capture_output=True, text=True)
        line = [l for l in res.stdout.splitlines() if l.startswith("cross-fitted")]
        print("%-22s p95 %.2f ms | %s" % (var, float(np.percentile(lat, 95)), line[0] if line else res.stderr[-300:]),
              flush=True)


if __name__ == "__main__":
    main()
