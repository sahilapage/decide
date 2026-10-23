"""
Run the hand-written, judge-style cases in tests/judge_cases.json: short,
hand-typed tickets like the ones a judge might type during the live demo.
Not a score; a sanity check that the model behaves sensibly off-template.

  python tools/judge_cases.py [--artifacts artifacts]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DecisionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--cases", default="tests/judge_cases.json")
    args = ap.parse_args()
    m = DecisionModel(args.artifacts)
    ok = n = 0
    for c in json.load(open(args.cases)):
        s = c["state"]
        if c["decision"] == "route":
            v, probs, conf, ab = m.route(s)
            p = conf
        else:
            v, p, ab = getattr(m, c["decision"])(s)
        good = v == c["expect"]
        ok += good
        n += 1
        print("%-4s %-8s expect %-9s got %-9s p %.3f %s %s  (%s)" % (
            c["id"], c["decision"], c["expect"], v, p, "abstain" if ab else "       ",
            "OK " if good else "XX ", c["why"]))
    print("\n%d / %d as expected" % (ok, n))


if __name__ == "__main__":
    main()
