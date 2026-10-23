"""
Offline copy of the selfcheck.py scoring maths, so we can score a model's raw
outputs in-process (no HTTP) while training and tuning.

It imports selfcheck's own metric functions and mirrors its main() exactly,
minus latency. Use it for iteration only; the official number always comes
from running selfcheck.py against the live service.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selfcheck as sc  # noqa: E402

QUEUES = sc.QUEUES


def score(records, outs, latency_score=1.0, verbose=False):
    """
    records: list of dataset rows (with labels)
    outs: {"route": [resp...], "approve": [...], "grounded": [...]} aligned with records,
          each resp shaped like the API body (value, probs|p_true, abstain).
    """
    def answered(r):
        return bool(r) and not r["abstain"]

    # route
    confs, corr, ans, pairs, prob_rows, truths = [], [], [], [], [], []
    for rec, resp in zip(records, outs["route"]):
        t = rec["labels"]["route"]
        probs = resp["probs"]
        top = max(probs, key=probs.get)
        confs.append(float(probs[top]))
        corr.append(1.0 if top == t else 0.0)
        prob_rows.append(probs)
        truths.append(t)
        ans.append(answered(resp))
        if answered(resp):
            pairs.append((t, resp["value"]))
    route_cov = sum(ans) / len(ans)
    route_f1 = sc.macro_f1(pairs, QUEUES)
    route_acc = sum(corr) / len(corr)
    route_ece = sc.ece(confs, corr)
    route_risk = sc.selective_risk(confs, corr)

    def binary(dec):
        ys = [1.0 if rec["labels"][dec] else 0.0 for rec in records]
        ps = [float(r["p_true"]) for r in outs[dec]]
        vs = [bool(r["value"]) for r in outs[dec]]
        an = [answered(r) for r in outs[dec]]
        bpairs = [(bool(y), v) for y, v, a in zip(ys, vs, an) if a]
        b_brier = sc.brier(ps, ys)
        b_ece = sc.ece(ps, ys)
        bconf = [max(p, 1 - p) for p in ps]
        bcorr = [1.0 if (p > 0.5) == (y > 0.5) else 0.0 for p, y in zip(ps, ys)]
        b_risk = sc.selective_risk(bconf, bcorr)
        return dict(ys=ys, ps=ps, vs=vs, an=an, pairs=bpairs, brier=b_brier, ece=b_ece,
                    risk=b_risk, cov=sum(an) / len(an), full_err=1 - sum(bcorr) / len(bcorr))

    A = binary("approve")
    G = binary("grounded")

    appr_cost = sc.cost_error([bool(y) for y, a in zip(A["ys"], A["an"]) if a],
                              [v for v, a in zip(A["vs"], A["an"]) if a])
    ans_ys = [y for y, a in zip(A["ys"], A["an"]) if a]
    rate = sum(ans_ys) / len(ans_ys) if ans_ys else 0.0
    hold_cost = rate
    gr_f1 = sc.macro_f1(G["pairs"], [True, False])

    ans_route = [t for t, a in zip(truths, ans) if a]
    maj = Counter(ans_route).most_common(1)[0][0] if ans_route else None
    triv_route_f1 = sc.macro_f1([(t, maj) for t in ans_route], QUEUES) if ans_route else 0.0
    ans_gys = [y for y, a in zip(G["ys"], G["an"]) if a]
    triv_gr_f1 = max(sc.macro_f1([(bool(y), True) for y in ans_gys], [True, False]),
                     sc.macro_f1([(bool(y), False) for y in ans_gys], [True, False])) if ans_gys else 0.0

    cs = lambda c: min(1.0, c / sc.TARGET_COVERAGE)
    q_route = sc.skill(route_f1, triv_route_f1) * cs(route_cov)
    q_appr = sc.cost_score(appr_cost, hold_cost) * cs(A["cov"])
    q_gr = sc.skill(gr_f1, triv_gr_f1) * cs(G["cov"])
    quality = (q_route + q_appr + q_gr) / 3

    mean_ece = (route_ece + A["ece"] + G["ece"]) / 3
    prior = {c: sum(1 for t in truths if t == c) / len(truths) for c in QUEUES}
    rb = sc.multiclass_brier(prob_rows, truths, QUEUES)
    rref = sc.multiclass_brier([prior] * len(truths), truths, QUEUES)
    ar = sum(A["ys"]) / len(A["ys"])
    gr = sum(G["ys"]) / len(G["ys"])
    bs = [sc.brier_skill(rb, rref), sc.brier_skill(A["brier"], ar * (1 - ar)),
          sc.brier_skill(G["brier"], gr * (1 - gr))]
    calibration = 0.5 * max(0.0, 1 - mean_ece / 0.20) + 0.5 * sum(bs) / 3

    full_err = [1 - route_acc, A["full_err"], G["full_err"]]
    ab = [max(0.0, 1 - r / e) if e > 1e-9 else 1.0
          for r, e in zip([route_risk, A["risk"], G["risk"]], full_err)]
    abstention = sum(ab) / 3

    parts = dict(quality=quality, calibration=calibration, latency=latency_score, abstention=abstention)
    total = sum(sc.WEIGHTS[k] * parts[k] for k in sc.WEIGHTS) * 100
    detail = dict(
        route_f1=route_f1, route_acc=route_acc, route_cov=route_cov, route_ece=route_ece,
        route_bskill=bs[0], route_risk=route_risk, q_route=q_route,
        appr_cost=appr_cost, appr_hold=hold_cost, appr_cov=A["cov"], appr_ece=A["ece"],
        appr_bskill=bs[1], appr_risk=A["risk"], appr_acc=1 - A["full_err"], q_appr=q_appr,
        gr_f1=gr_f1, gr_acc=1 - G["full_err"], gr_cov=G["cov"], gr_ece=G["ece"],
        gr_bskill=bs[2], gr_risk=G["risk"], q_gr=q_gr,
        ab_route=ab[0], ab_appr=ab[1], ab_gr=ab[2],
    )
    if verbose:
        print("  TOTAL %.2f | quality %.4f calib %.4f latency %.2f abstention %.4f"
              % (total, quality, calibration, latency_score, abstention))
        print("  route  acc %.4f F1 %.4f cov %.3f ECE %.4f Bskill %.3f risk %.4f ab %.3f"
              % (route_acc, route_f1, route_cov, route_ece, bs[0], route_risk, ab[0]))
        print("  approve acc %.4f cost %.4f (hold %.4f) q %.3f cov %.3f ECE %.4f Bskill %.3f risk %.4f ab %.3f"
              % (1 - A["full_err"], appr_cost, hold_cost, q_appr, A["cov"], A["ece"], bs[1], A["risk"], ab[1]))
        print("  ground acc %.4f F1 %.4f cov %.3f ECE %.4f Bskill %.3f risk %.4f ab %.3f"
              % (1 - G["full_err"], gr_f1, G["cov"], G["ece"], bs[2], G["risk"], ab[2]))
    return total, parts, detail
