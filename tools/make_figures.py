"""
Presentation figures, 16:9 PNGs in docs/figures/.

  pip install matplotlib          # only needed for this script
  python tools/make_figures.py

Numbers come from the repo's own result files where they exist
(docs/stress/*.json, docs/ceilings.json, data/*.jsonl). The rest are the
measured values recorded in the docs, each cited next to its constant below.
"""

import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

# validated categorical palette (light mode), fixed slot order, never cycled
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 12,
})


def figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_title(title)
    ax.text(0.0, -0.13, textwrap.fill(subtitle, 125), color=MUTED, fontsize=11, transform=ax.transAxes,
            va="top", ha="left")
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
def score_vs_ceiling():
    # docs/ceiling.md: shipped model vs the best any model can reach on this data
    parts = [("Latency (20)", 20.0, 20.0), ("Quality (40)", 34.3, 34.6),
             ("Calibration (25)", 21.9, 22.1), ("Abstention (15)", 9.7, 10.0)]
    fig, ax = figure("We are within ~1 point of the data's ceiling",
                     "Points per score part (weight in brackets). Ceiling estimated from out-of-fold "
                     "predictions and label-noise analysis (docs/ceiling.md).")
    y = np.arange(len(parts))[::-1]
    for yi, (name, ours, best) in zip(y, parts):
        ax.barh(yi, ours, height=0.42, color=BLUE, zorder=2)
        ax.plot([best, best], [yi - 0.32, yi + 0.32], color=INK, lw=2.5, zorder=3)
        ax.text(ours - 0.4, yi, "%.1f" % ours, va="center", ha="right", color="white", fontsize=13,
                fontweight="bold", zorder=4)
        ax.text(best + 0.4, yi, "max %.1f" % best, va="center", ha="left", color=INK2, fontsize=12)
    ax.set_yticks(y, [p[0] for p in parts], fontsize=14, color=INK)
    ax.set_xlim(0, 42)
    ax.set_xlabel("points")
    ax.grid(axis="y", visible=False)
    ax.text(41.5, 3.0, "Total: 85.9\nof a possible ~87", ha="right", va="center", fontsize=17,
            fontweight="bold", color=INK, linespacing=1.4)
    save(fig, "1_score_vs_ceiling.png")


def score_progression():
    # dev selfcheck at each milestone (docs/team/context.md section 6, README, commit messages)
    steps = [("Gate-1\nn-gram baseline", 77.0), ("Ettin-32M\nmulti-task", 81.95),
             ("+ normaliser,\nflags, calibration", 84.6), ("17M + short\nprompts", 85.22),
             ("+ robustness\naugmentation", 84.98), ("+ approve\nsecond stage", 85.87)]
    fig, ax = figure("From 77 to 85.9 on the judges' scorer",
                     "selfcheck.py total on the dev split at each milestone. The robustness step "
                     "trades ~0.2 on clean dev for +5.5 on shifted text (see the robustness figure).")
    x = np.arange(len(steps))
    v = [s[1] for s in steps]
    ax.axhline(87, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(len(steps) - 1 + 0.3, 87.25, "data ceiling ~87", ha="right", va="bottom", color=INK2, fontsize=12)
    ax.plot(x, v, color=BLUE, lw=2.5, zorder=2)
    ax.scatter(x, v, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for xi, vi in zip(x, v):
        ax.text(xi, vi - 0.7, "%.1f" % vi, ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks(x, [s[0] for s in steps], fontsize=12, color=INK)
    ax.set_ylim(75, 89)
    ax.set_ylabel("total score / 100")
    ax.grid(axis="x", visible=False)
    save(fig, "2_score_progression.png")


def latency():
    # docs/benchmarks.md: mean-of-decisions p95, in-process, i9-13900K, 2 physical cores (4 vCPU),
    # cross-fitted dev score of each variant
    rows = [("Ettin-32M fp32", 14.2, "dev score 83.4 (cross-fitted)"), ("Ettin-32M int8", 7.9, "dev score 80.4: −3 points, grounded 99% → 92%"),
            ("Ettin-17M fp32  (shipped)", 5.4, "same quality as 32M")]
    fig, ax = figure("Shipped model: ~5 ms p95 on 2 CPU cores, 5× under the full-marks line",
                     "p95 latency, mean of the three decisions, in-process, i9-13900K pinned to 2 physical "
                     "cores (docs/benchmarks.md). Over HTTP add ~1-3 ms. The 100 ms gate is off the chart.")
    y = np.arange(len(rows))[::-1]
    for yi, (name, ms, note) in zip(y, rows):
        col = BLUE if "shipped" in name else "#9ec5f4"
        ax.barh(yi, ms, height=0.42, color=col, zorder=2)
        ax.text(ms + 0.4, yi + 0.02, "%.1f ms" % ms, va="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(ms + 0.4, yi - 0.28, note, va="center", fontsize=11, color=INK2)
    ax.axvline(25, color=INK, lw=2)
    ax.text(24.6, 2.42, "full marks ≤ 25 ms", ha="right", va="center", fontsize=12, color=INK)
    ax.set_ylim(-0.6, 2.62)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14, color=INK)
    ax.set_xlim(0, 30)
    ax.set_xlabel("p95 latency (ms)")
    ax.grid(axis="y", visible=False)
    save(fig, "3_latency.png")


def robustness():
    old = {r["variant"]: r["total"] for r in load_json("docs", "stress", "shipped_v5h_s3.json")}
    new = {r["variant"]: r["total"] for r in load_json("docs", "stress", "v6t2h_s3.json")}
    names = {"clean": "clean dev", "typos": "typos", "lower_nopunct": "lowercase, no punctuation",
             "synonyms": "synonyms", "body_shuffle": "ticket sentences reordered",
             "amount_format": "\"$12\" → \"12 USD\"", "kb_shuffle": "KB sentences reordered",
             "draft_shuffle": "draft sentences reordered", "kb_distractor": "extra unrelated KB sentence",
             "num_words": "digits ↔ number words", "draft_ticket_ref": "draft quotes order number",
             "all_mixed": "3 shifts at once", "num_flip": "one number changed (label flips)"}
    keys = [k for k in names if k in old and k in new]
    keys.sort(key=lambda k: old[k])
    fig, ax = figure("Robustness: the worst case went from 65 to 79",
                     "Total score on dev tickets rewritten in label-preserving ways (tools/stress_test.py). "
                     "Original model vs shipped recipe; same seed, no refit on the shifted data.")
    y = np.arange(len(keys))
    for yi, k in zip(y, keys):
        ax.plot([old[k], new[k]], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([old[k] for k in keys], y, s=80, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="original model (no augmentation)")
    ax.scatter([new[k] for k in keys], y, s=80, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="shipped recipe (grounded-only augmentation)")
    for yi, k in zip(y, keys):
        if new[k] - old[k] > 3:
            ax.text(new[k] + 0.5, yi, "+%.0f" % (new[k] - old[k]), va="center", fontsize=11,
                    color=INK2, fontweight="bold")
    ax.set_yticks(y, [names[k] for k in keys], fontsize=12, color=INK)
    ax.set_xlim(62, 89)
    ax.set_xlabel("total score / 100")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, "4_robustness.png")


def ceiling_curves():
    c = load_json("docs", "ceilings.json")
    fig, ax = figure("Route errors are label noise: even our surest 10% is 5.5% wrong",
                     "Error rate among the k% most confident out-of-fold predictions (6,500 train tickets, "
                     "tools/ceilings.py). A flat curve means no confidence signal can find those errors.")
    for key, col, lab in [("route", BLUE, "route"), ("approve", ORANGE, "approve")]:
        cur = c[key]["curve"]
        ks = sorted(cur, key=float)
        x = [100 * float(k) for k in ks]
        yv = [100 * cur[k] for k in ks]
        ax.plot(x, yv, color=col, lw=2.5, zorder=2)
        ax.scatter(x, yv, s=60, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(x[-1] + 2, yv[-1], lab, color=INK, fontsize=13, fontweight="bold", va="center")
    ax.axhline(5.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(12, 6.0, "5.5% random route labels: no model can rank these", color=INK2, fontsize=12)
    ax.set_xlim(5, 112)
    ax.set_ylim(0, 13)
    ax.set_xlabel("most confident k% of tickets")
    ax.set_ylabel("error rate (%)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d%%" % v))
    save(fig, "5_noise_ceiling.png")


def approve_soft_cap():
    rows = []
    for f in ("train.jsonl", "dev.jsonl"):
        p = os.path.join(ROOT, "data", f)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if not rows:
        print("skip approve_soft_cap: data/*.jsonl not found")
        return
    # approximate centre of each tier's soft cap (docs/team/context.md section 3)
    cap = {"free": 25.0, "pro": 110.0, "enterprise": 390.0}
    edges = np.arange(0.0, 1.75, 0.125)
    fig, ax = figure("Approve: the amount cap is soft, so near it the label is close to a coin flip",
                     "Share of requests approved vs amount ÷ the tier's cap (8,000 train+dev tickets, amount > 0). "
                     "Cap centres: free \\$25, pro \\$110, enterprise \\$390.")
    for tier, col in [("free", BLUE), ("pro", ORANGE), ("enterprise", AQUA)]:
        r = [(x["state"]["amount"] / cap[tier], x["labels"]["approve"]) for x in rows
             if x["state"]["customer_tier"] == tier and x["state"]["amount"] > 0]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = [a for v, a in r if lo <= v < hi]
            if len(b) >= 15:
                xs.append((lo + hi) / 2)
                ys.append(100 * np.mean(b))
        ax.plot(xs, ys, color=col, lw=2.5, zorder=2)
        ax.scatter(xs, ys, s=50, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        # label beside the last point still above ~0, so tiers that all end at 0% don't stack
        k = max(i for i, yv in enumerate(ys) if yv > 2)
        ax.text(xs[k] + 0.03, ys[k] + 3, tier, color=INK, fontsize=13, fontweight="bold", va="center")
    ax.axvspan(0.8, 1.1, color=GRID, alpha=0.6, zorder=0)
    ax.text(0.95, 92, "soft-cap zone", ha="center", color=INK2, fontsize=12)
    ax.axhline(80, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.18, 84, "~15-20% refused for hidden account\nhistory, even far under the cap",
            color=INK2, fontsize=11, va="bottom")
    ax.set_xlim(0, 1.85)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("amount ÷ tier cap")
    ax.set_ylabel("approved (%)")
    save(fig, "6_approve_soft_cap.png")


def extensibility():
    # docs/extensibility.md: grounded added to a route+approve model, dev total (cross-fitted)
    rows = [("all 3 trained together (reference)", 83.61, 160, MUTED),
            ("add grounded later: fine-tune + replay, all labels", 83.19, 89, BLUE),
            ("add grounded later: 1,000 labels only", 75.04, 59, "#9ec5f4"),
            ("add grounded later: frozen encoder, new head only", 72.90, 77, "#9ec5f4"),
            ("route + approve only (before adding)", 72.21, 75, GRID)]
    fig, ax = figure("A 4th decision is cheap: added later in 89 s, it matches joint training",
                     "Leave-one-out test: train on route + approve, then add grounded as if new (one prompt, "
                     "one head). Dev total, cross-fitted (±1.1 noise). docs/extensibility.md")
    y = np.arange(len(rows))[::-1]
    for yi, (name, v, sec, col) in zip(y, rows):
        ax.barh(yi, v, height=0.5, color=col, zorder=2)
        ax.text(v + 0.3, yi, "%.1f   (%d s)" % (v, sec), va="center", fontsize=13, color=INK,
                fontweight="bold" if col == BLUE else "normal")
    ax.axvspan(83.61 - 1.1, 83.61 + 1.1, color=GRID, alpha=0.5, zorder=0)
    ax.text(83.61, -0.75, "joint ±1.1 noise", ha="center", color=INK2, fontsize=11)
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12, color=INK)
    ax.set_xlim(65, 92)
    ax.set_xlabel("dev total / 100")
    ax.grid(axis="y", visible=False)
    save(fig, "7_extensibility.png")


def losses():
    # loss experiment on Ettin-17M: 5-fold out-of-fold total on 6,500 train tickets, same seed
    rows = [("base (cross-entropy)", 84.43), ("L2-SP anchor", 84.32), ("layer-wise LR decay", 84.09),
            ("R-Drop consistency", 84.05), ("attention pooling", 83.12), ("focal loss", 81.97),
            ("GCE (noise-robust)", 80.68)]
    fig, ax = figure("Six loss and architecture tweaks: none beats plain training",
                     "Out-of-fold total on 6,500 train tickets, Ettin-17M, same seed, no augmentation or tree stage. The ones "
                     "that lost did so on grounded confidence, not accuracy. A second run on the shipped recipe "
                     "(augmentation + tree stage, docs/report.md #21-22) found GCE +0.24 and GCE + LR decay +0.29, "
                     "about 1 sd: not enough to ship either.")
    y = np.arange(len(rows))[::-1]
    base = rows[0][1]
    ax.axvspan(base - 0.5, base + 0.5, color=GRID, alpha=0.6, zorder=0)
    ax.text(base, -0.75, "±0.5 noise band", ha="center", color=INK2, fontsize=11)
    ax.set_ylim(-1.0, len(rows) - 0.5)
    for yi, (name, v) in zip(y, rows):
        col = BLUE if name.startswith("base") else (ORANGE if v < base - 0.5 else MUTED)
        ax.plot([79.5, v], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter([v], [yi], s=110, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(v + 0.25, yi, "%.2f" % v, va="center", fontsize=12, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=13, color=INK)
    ax.set_xlim(79.5, 86)
    ax.set_xlabel("out-of-fold total / 100")
    ax.grid(axis="y", visible=False)
    save(fig, "8_losses.png")


def complex_queries():
    # shipped model with and without the long-KB focus (decide/textfmt.py focus_kb), same rewritten dev
    # sets from tools/stress_test.py --only ...; docs/robustness.md round 3
    rows = [("clean dev", 85.74, 85.74), ("ticket inside an email thread", 85.74, 85.74),
            ("longer ticket body", 86.54, 86.54), ("one extra KB sentence", 84.30, 84.30),
            ("two issues in one ticket", 83.72, 83.72), ("long KB article (7-9 sentences)", 78.57, 81.44),
            ("all of the above at once", 76.35, 80.89)]
    fig, ax = figure("Complex queries: long KB articles fixed at inference, nothing else moves",
                     "Shipped model on dev tickets rewritten to be longer and messier (label-preserving). "
                     "The fix only activates for KB articles longer than any in training, so clean inputs "
                     "are byte-identical (selfcheck unchanged at 85.87; 85.74 here only because calls run in dataset order, not selfcheck's shuffled order).")
    y = np.arange(len(rows))[::-1]
    for yi, (name, a, b) in zip(y, rows):
        ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([r[1] for r in rows], y, s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="without the fix")
    ax.scatter([r[2] for r in rows], y, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4,
               label="with long-KB focus")
    for yi, (name, a, b) in zip(y, rows):
        txt = "+%.1f" % (b - a) if b - a > 0.05 else "unchanged"
        ax.text(max(a, b) + 0.35, yi, txt, va="center", fontsize=12, color=INK2,
                fontweight="bold" if b - a > 0.05 else "normal")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=13, color=INK)
    ax.set_xlim(74, 90)
    ax.set_xlabel("total score / 100")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "9_complex_queries.png")


def training_data():
    # same shipped pipeline (augmentation, seed 3, approve stage, calibration) on 5,000 vs all 6,500
    # train tickets, scored on the same dev split, policy cross-fitted over 10 halves
    rows = [("5,000 train tickets", 84.26), ("6,500 train tickets\n(the shipped recipe)", 85.05)]
    fig, ax = figure("More training data still helps: +0.8 for 1,500 more tickets",
                     "Full shipped pipeline retrained from scratch on each training size; dev total, cross-fitted. "
                     "One seed per size, so treat +0.8 as indicative. The 6,500 run reproduces the shipped "
                     "model (85.1).")
    x = np.arange(len(rows))
    for xi, (name, v) in zip(x, rows):
        ax.bar(xi, v - 80, bottom=80, width=0.45, color=BLUE if xi == 1 else "#9ec5f4", zorder=2)
        ax.text(xi, v + 0.12, "%.2f" % v, ha="center", va="bottom", fontsize=16, fontweight="bold", color=INK)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=14, color=INK)
    ax.set_ylim(80, 87)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("dev total / 100 (axis starts at 80)")
    ax.grid(axis="x", visible=False)
    ax.annotate("", xy=(1, 85.6), xytext=(0, 84.8), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
    ax.text(0.5, 85.55, "+0.79", ha="center", va="bottom", fontsize=14, color=INK2, fontweight="bold")
    save(fig, "10_training_data.png")


def approve_stage():
    # Sahil's approve second stage (decide/approve_stage.py): 60 boosted trees on the network's approve
    # logit + amount/cap + phrase flags, fitted on out-of-fold logits, compiled into the ONNX graph.
    # Numbers from commit 27c82a6 and README.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=160)
    fig.suptitle("Approve second stage: +1.8 points of approve accuracy, +1.0 on the total",
                 x=0.06, ha="left", fontsize=20, fontweight="bold", color=INK)
    panels = [(a1, "approve accuracy, out-of-fold (%)", [88.5, 90.3], 86, 92, "%.1f%%"),
              (a2, "total score, dev, cross-fitted", [84.1, 85.1], 82, 87, "%.1f")]
    for ax, label, vals, lo, hi, fmt in panels:
        ax.set_axisbelow(True)
        for xi, v in enumerate(vals):
            ax.bar(xi, v - lo, bottom=lo, width=0.5, color=BLUE if xi else "#9ec5f4", zorder=2)
            ax.text(xi, v + (hi - lo) * 0.015, fmt % v, ha="center", va="bottom", fontsize=16,
                    fontweight="bold", color=INK)
        ax.set_xticks([0, 1], ["network only", "+ tree stage"], fontsize=13, color=INK)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(label, fontsize=14, fontweight="normal", color=INK2, loc="left")
        ax.grid(axis="x", visible=False)
    fig.text(0.06, -0.02, textwrap.fill(
        "60 boosted trees refine the shared network's approve logit using amount ÷ tier cap and the "
        "disqualifying-phrase flags. Fitted on out-of-fold logits and compiled into the same ONNX graph "
        "(still one model, ~0.1 ms). Axes start above zero to show the change.", 125),
        color=MUTED, fontsize=11, va="top")
    fig.subplots_adjust(top=0.84, wspace=0.25, left=0.06, right=0.98)
    save(fig, "11_approve_stage.png")


# ---------------------------------------------------------------------------
# Figures 12-15. 13-15 plot docs/figures/data/model_dev.json, written by
# tools/figure_data.py from the shipped model's own answers on dev.

def architecture():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6.6), dpi=160)
    ax.set_xlim(-1.5, 131.5)
    ax.set_ylim(0, 66)
    ax.axis("off")
    ax.text(0, 64, "One request, one forward pass, one typed answer", fontsize=20, fontweight="bold", va="top")
    ax.text(0, 59.6, "Everything inside the dashed box is a single ONNX graph (artifacts/student/model.onnx).",
            fontsize=12, color=INK2, va="top")

    def box(x, y, w, h, title, body, fill="#ffffff", edge="#d9d7d1", tc=INK, bc=INK2, tag=None, tagc=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                    fc=fill, ec=edge, lw=1.3, zorder=2))
        top = y + h - 1.4
        if tag:
            ax.text(x + 1.1, top, tag, fontsize=9, color=tagc, fontweight="bold", va="top", zorder=3,
                    family="DejaVu Sans Mono")
            top -= 3.0
        ax.text(x + 1.1, top, title, fontsize=12.5, fontweight="bold", color=tc, va="top", zorder=3)
        ax.text(x + 1.1, top - 3.4, body, fontsize=10, color=bc, va="top", zorder=3, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, shrinkA=0, shrinkB=0))

    mid = 31
    box(0, 16, 19, 30, "Request", "decision:\nroute, approve\nor grounded\n\nticket: subject,\nbody, tier, amount\n\ndraft reply +\nKB article")
    box(23, 16, 22, 30, "Clean + check", "\"two weeks\" = \"14 days\"\ntypo repair, ranges\n\nfact flags: numbers\nor scope the KB\ndoes not state\n\nlong KB: keep the\nrelevant sentences",
        tag="OURS")
    ax.add_patch(FancyBboxPatch((48.5, 4), 54, 49, boxstyle="round,pad=0.3,rounding_size=2",
                                fc="none", ec=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1))
    ax.text(49.5, 51.8, "one ONNX graph", fontsize=9.5, color=MUTED, va="top", family="DejaVu Sans Mono")
    box(50.5, 16, 21, 30, "Ettin-17M", "shared encoder:\none set of weights\nreads all three\nquestions\n\na typed prompt\nnames the question\n\nopen source (MIT),\nfine-tuned by us",
        fill=INK, edge=INK, tc="#ffffff", bc="#c9ced6", tag="SHARED", tagc="#f0a15e")
    heads = [(37.5, 10, "Route head", "linear layer -> 5 queues"),
             (21.5, 13, "Approve head", "small MLP + 60 boosted\ntrees (amount / tier cap,\nrefusal phrases)"),
             (6.5, 11.5, "Grounded head", "small MLP + learned\nweight on fact flags")]
    for y, h, t, b in heads:
        box(76, y, 24.5, h, t, b)
        arrow(72.3, mid, 75.6, y + h / 2)
    box(106, 16, 24, 30, "Calibrate + abstain", "noise-aware softmax;\nPlatt with floor and\nceiling\n\napprove only if\np > 0.75 (3:1 cost)\n\nabstain when unsure",
        tag="OURS")
    arrow(19.6, mid, 22.6, mid)
    arrow(45.6, mid, 50.1, mid)
    for y, h, _, _ in heads:
        arrow(101.1, y + h / 2, 105.6, mid)
    ax.text(106, 13.5, "-> {value, p, abstain}", fontsize=11, color=INK, family="DejaVu Sans Mono", va="top")
    ax.text(0, 1.2, "~6 ms p95 on 4 CPU threads  ·  68 MB  ·  no network, no GPU  ·  a 4th decision = one prompt + one head",
            fontsize=11.5, color=INK2)
    save(fig, "12_architecture.png")


def _model_data():
    return load_json("docs", "figures", "data", "model_dev.json")


def calibration():
    d = _model_data()
    fig, ax = figure("Calibrated: when it says 80%, it is right about 80% of the time",
                     "Shipped model on the 1,500 dev tickets, bins 0.1 wide (bins under 15 tickets left out); dot size = tickets per bin. "
                     "Route: top-queue confidence vs accuracy. Approve: p(approve) vs share actually approved. "
                     "Grounded is 100% correct with p ≈ 0 or 1 (ECE 0.0001), so it is left off.")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--", zorder=1)
    ax.text(0.30, 0.25, "perfect calibration", color=MUTED, fontsize=11, rotation=33, ha="center")
    rc = json.load(open(os.path.join(ROOT, "results.json")))["calibration"]
    series = [("route", BLUE, [(c, ok) for c, ok, _ in d["route"]], rc["route_ece"]),
              ("approve", ORANGE, [(p, y) for p, y, _, _ in d["approve"]], rc["approve_ece"])]
    for name, col, pairs, ece in series:
        a = np.array(pairs, dtype=float)
        idx = np.minimum((a[:, 0] * 10).astype(int), 9)
        bins = [a[idx == k] for k in range(10) if (idx == k).sum() >= 15]
        xs = [b[:, 0].mean() for b in bins]
        ys = [b[:, 1].mean() for b in bins]
        ax.plot(xs, ys, color=col, lw=2.2, zorder=2)
        ax.scatter(xs, ys, s=[len(b) * 0.9 for b in bins], color=col, edgecolor=SURFACE, lw=1.5, zorder=3,
                   label="%s  (ECE %.3f)" % (name, ece))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(loc="upper left", fontsize=13)
    ax.text(0.98, 0.62, "route tops out near 0.95: ~5% of\nroute labels are random, so no\nticket is ever 100% sure\n\napprove tops out near 0.85: some\nsmall refunds are refused for\nhistory the ticket never shows",
            ha="right", va="top", fontsize=11, color=INK2)
    save(fig, "13_calibration.png")


def coverage():
    d = _model_data()
    fig, ax = figure("It knows which answers to trust: fewer answers, fewer mistakes",
                     "Accuracy on the k% of dev tickets the model is most confident about (route: top-queue "
                     "probability; approve: distance of p from 0.5). Dots = where the shipped abstain policy "
                     "operates; the judges' quality target is 80% coverage.")
    series = [("route", BLUE, [(c, ok, ab) for c, ok, ab in d["route"]]),
              ("approve", ORANGE, [(abs(p - 0.5), ok, ab) for p, _, ok, ab in d["approve"]])]
    ks = np.arange(10, 101)
    lines = []
    for name, col, rows in series:
        rows.sort(key=lambda r: -r[0])
        ok = np.array([r[1] for r in rows], dtype=float)
        acc = [ok[: max(1, int(round(len(ok) * k / 100)))].mean() * 100 for k in ks]
        ax.plot(ks, acc, color=col, lw=2.4, label=name)
        cov = 100 * (1 - np.mean([r[2] for r in rows]))
        ans = [r[1] for r in rows if not r[2]]
        ax.scatter([cov], [100 * np.mean(ans)], s=130, color=col, edgecolor=INK, lw=1.2, zorder=4)
        dy = 2.2 if name == "route" else -2.6
        ax.annotate("shipped %s: answers %.0f%%,\n%.1f%% of those right" % (name, cov, 100 * np.mean(ans)),
                    xy=(cov, 100 * np.mean(ans)), xytext=(cov - 23, 100 * np.mean(ans) + dy), fontsize=11,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
        lines.append("%s %.1f%%" % (name, acc[-1]))
    ax.axvline(80, color=MUTED, lw=1.3, ls="--")
    ax.text(80.6, 99.6, "80% target", color=MUTED, fontsize=11, va="top")
    ax.text(99, 86.0, "answer everything:\n" + ", ".join(lines), ha="right", fontsize=11, color=INK2)
    ax.set_xlim(10, 101)
    ax.set_ylim(85, 100.5)
    ax.set_xlabel("share of tickets answered, most confident first (%)")
    ax.set_ylabel("accuracy on answered tickets (%)")
    ax.legend(loc="lower left", fontsize=13)
    save(fig, "14_coverage.png")


def approve_curve():
    d = _model_data()
    caps = {"free": 25, "pro": 110, "enterprise": 385}
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.75), dpi=160, sharey=True)
    fig.suptitle("Approve: the same refund on every plan, swept over the amount",
                 x=0.06, ha="left", fontsize=20, fontweight="bold", color=INK)
    for ax, (tier, col) in zip(axes, [("free", BLUE), ("pro", ORANGE), ("enterprise", AQUA)]):
        pts = np.array(d["sweep"][tier], dtype=float)
        amt, p, ab = pts[:, 0], pts[:, 1], pts[:, 3]
        on = False
        for i in range(len(amt)):
            if ab[i] and not on:
                start, on = amt[i], True
            if on and (not ab[i] or i == len(amt) - 1):
                ax.axvspan(start, amt[i], color=GRID, alpha=0.9, zorder=0)
                on = False
        ax.plot(amt, p, color=col, lw=2.6, zorder=3)
        ax.axhline(0.75, color=INK, lw=1.2, ls="--", zorder=2)
        ax.axvline(caps[tier], color=MUTED, lw=1, ls=":", zorder=2)
        ax.text(caps[tier], 0.93 if tier == "enterprise" else 1.0, " cap ~$%d" % caps[tier], color=INK2,
                fontsize=10.5, va="top")
        ax.set_title(tier, fontsize=15, fontweight="bold", color=col, loc="left")
        ax.set_xlabel("refund amount ($)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, amt[-1])
        ax.grid(axis="x", visible=False)
    ent = np.array(d["sweep"]["enterprise"], dtype=float)
    k = int(np.argmin(np.where(ent[:, 0] < 380, ent[:, 1], 9)))
    if ent[k, 1] < 0.5:
        axes[2].annotate("dip at $%.2f: known weak spot\n(\"$200\" tickets in training were\nall refused; see model card)" % ent[k, 0],
                         xy=(ent[k, 0], ent[k, 1]), xytext=(15, 1.04), fontsize=10, color=INK2, va="top",
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    axes[0].set_ylabel("p(approve), calibrated")
    axes[0].text(1, 0.77, "approve if p > 0.75", fontsize=10.5, color=INK, va="bottom")
    fig.text(0.06, -0.02, textwrap.fill(
        "Ticket: \"Please refund my last charge, I did not use it.\", amounts with cents as in the data. Shaded = the model abstains and hands the "
        "case to a human. A wrong approval costs 3x a wrong hold, so approving pays only above p = 0.75. "
        "About 15-20% of small refunds are refused in the data for account history the ticket never shows, "
        "so p never goes above ~0.85.", 125), color=MUTED, fontsize=11, va="top")
    fig.subplots_adjust(top=0.84, wspace=0.12, left=0.07, right=0.98)
    save(fig, "15_approve_curve.png")


if __name__ == "__main__":
    architecture()
    calibration()
    coverage()
    approve_curve()
    score_vs_ceiling()
    score_progression()
    latency()
    robustness()
    ceiling_curves()
    approve_soft_cap()
    extensibility()
    losses()
    complex_queries()
    training_data()
    approve_stage()
