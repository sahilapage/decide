"""
One table from several stress-test result files (tools/stress_test.py --out).

  python tools/compare_stress.py docs/stress/*.json

"shift mean" = mean total over the label-preserving shifted variants (not clean,
not all_mixed, not num_flip). "held-out mean" = the same over variants no
training augmentation family touches.
"""

import json
import os
import sys

HELD_OUT = {"lower_nopunct", "synonyms", "body_shuffle", "amount_format", "draft_shuffle", "num_words",
            "draft_ticket_ref"}


def main():
    files = sys.argv[1:]
    rows = {os.path.basename(f).replace("stress_", "").replace(".json", ""): json.load(open(f)) for f in files}
    variants = []
    for r in rows.values():
        for v in r:
            if v["variant"] not in variants:
                variants.append(v["variant"])
    names = list(rows)
    w = max(len(n) for n in names) + 2
    print("%-17s" % "variant" + "".join(n.rjust(w) for n in names))
    for v in variants:
        line = "%-17s" % v
        for n in names:
            m = {x["variant"]: x for x in rows[n]}
            line += ("%.2f" % m[v]["total"]).rjust(w) if v in m else "-".rjust(w)
        print(line)
    for label, pick in [("shift mean", lambda v: v not in ("clean", "all_mixed", "num_flip")),
                        ("held-out mean", lambda v: v in HELD_OUT)]:
        line = "%-17s" % label
        for n in names:
            vals = [x["total"] for x in rows[n] if pick(x["variant"])]
            line += ("%.2f" % (sum(vals) / len(vals))).rjust(w) if vals else "-".rjust(w)
        print(line)
    print("\ngrounded accuracy")
    for v in variants:
        line = "%-17s" % v
        for n in names:
            m = {x["variant"]: x for x in rows[n]}
            line += ("%.3f" % m[v]["grounded_acc"]).rjust(w) if v in m else "-".rjust(w)
        print(line)


if __name__ == "__main__":
    main()
