"""
Repeat bench_cpu.py over CPU layouts x ONNX variant x threads x spinning and
report the median of each config's mean-p95, so one noisy run cannot decide.

  python tools/latency_sweep.py exports/e17 results.csv [--reps 3]

Layouts pin the process with taskset. "4phys" is four distinct physical cores;
"2phys_4vcpu" is four logical CPUs on two physical cores (a typical 4-vCPU VM).
Check `lscpu -e=CPU,CORE` and edit LAYOUTS if your core numbering differs.
Spinning is toggled with DECIDE_SPIN (see decide/student_runtime.py).
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

LAYOUTS = {"4phys": "0,2,4,6", "2phys_4vcpu": "0-3"}
# (variant, threads, spin)
PLANS = [("model.int8.onnx", "1,2,4", "1"), ("model.int8.onnx", "1,2,4", "0"),
         ("model.onnx", "4", "1")]
LINE = re.compile(r"^(\S+)\s+threads (\d+) \| (.*) \| mean p95 ([\d.]+) ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="export folder with model.onnx / model.int8.onnx")
    ap.add_argument("out_csv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rows = {}
    for rep in range(args.reps):
        for lname, cpus in LAYOUTS.items():
            for var, threads, spin in PLANS:
                env = dict(os.environ, DECIDE_SPIN=spin)
                cmd = ["taskset", "-c", cpus, sys.executable, "bench_cpu.py", args.folder,
                       "--n", str(args.n), "--variants", var, "--threads", threads]
                out = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True).stdout
                for line in out.splitlines():
                    m = LINE.match(line)
                    if not m:
                        continue
                    p50 = statistics.mean(float(x) for x in re.findall(r"p50 ([\d.]+)", m.group(3)))
                    key = (lname, m.group(1), int(m.group(2)), spin)
                    rows.setdefault(key, []).append((float(m.group(4)), p50))
        print("rep %d done" % rep, flush=True)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layout", "variant", "threads", "spin", "median_mean_p95", "min", "max", "median_p50"])
        for key in sorted(rows):
            p95 = [r[0] for r in rows[key]]
            p50 = [r[1] for r in rows[key]]
            w.writerow(list(key) + ["%.2f" % statistics.median(p95), "%.2f" % min(p95),
                                    "%.2f" % max(p95), "%.2f" % statistics.median(p50)])
    print(open(args.out_csv).read())


if __name__ == "__main__":
    main()
