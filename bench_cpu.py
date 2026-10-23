"""
In-process CPU latency + parity check for an exported student.

  python bench_cpu.py artifacts/student [--n 300] [--variants model.onnx,model.int8.onnx] [--threads 4]

Times one request at a time (no batching) per decision type, after warm-up,
and checks ONNX logits against the PyTorch dev logits if dev_logits.npz is
present in the folder.
"""

import argparse
import json
import os
import time

import numpy as np

from decide.student_runtime import StudentRuntime


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--variants", default="model.onnx,model.int8.onnx")
    ap.add_argument("--threads", default="4")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open("data/dev.jsonl")][: args.n]
    ref = None
    rp = os.path.join(args.folder, "dev_logits.npz")
    if os.path.exists(rp):
        ref = np.load(rp)
    for var in args.variants.split(","):
        if not os.path.exists(os.path.join(args.folder, var)):
            continue
        for th in args.threads.split(","):
            os.environ["DECIDE_ONNX"] = var
            os.environ["DECIDE_THREADS"] = th
            rt = StudentRuntime(args.folder)
            for r in rows[:30]:
                for d in ("route", "approve", "grounded"):
                    rt.logits(d, r["state"])
            lat = {d: [] for d in ("route", "approve", "grounded")}
            out = {d: [] for d in lat}
            for r in rows:
                for d in lat:
                    t0 = time.perf_counter()
                    z = rt.logits(d, r["state"])
                    lat[d].append((time.perf_counter() - t0) * 1000)
                    out[d].append(z)
            msg = "  ".join("%s p50 %.2f p95 %.2f" % (d, pct(v, 50), pct(v, 95)) for d, v in lat.items())
            print("%-18s threads %s | %s | mean p95 %.2f ms"
                  % (var, th, msg, np.mean([pct(v, 95) for v in lat.values()])))
            if ref is not None:
                n = len(rows)
                rr = np.asarray(out["route"])
                ra = ref["route"][:n] - np.log(np.exp(ref["route"][:n]).sum(1, keepdims=True))
                print("    parity  route argmax agree %.4f  max|dlogp| %.3f | approve max|d| %.3f sign agree %.4f | grounded max|d| %.3f sign agree %.4f"
                      % ((rr.argmax(1) == ra.argmax(1)).mean(), np.abs(rr - ra).max(),
                         np.abs(np.asarray(out["approve"]) - ref["approve"][:n]).max(),
                         ((np.asarray(out["approve"]) > 0) == (ref["approve"][:n] > 0)).mean(),
                         np.abs(np.asarray(out["grounded"]) - ref["grounded"][:n]).max(),
                         ((np.asarray(out["grounded"]) > 0) == (ref["grounded"][:n] > 0)).mean()))


if __name__ == "__main__":
    main()
