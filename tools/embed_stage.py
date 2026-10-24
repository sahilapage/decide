"""
Compile the approve second stage (approve_stage.json, 60 boosted trees) INTO
the model's ONNX graph, so the whole system is one graph with one forward
pass and three outputs: the trees read the graph's own approve logit plus a
small `stage_x` input (tier, amount / tier cap, amount == 0, phrase flags).

  python tools/embed_stage.py exports/v6t2h_s3_st          # rewrites model.onnx in place
  python tools/embed_stage.py exports/x --check data/dev.jsonl

Uses the standard ai.onnx.ml TreeEnsembleRegressor operator, which ONNX
Runtime runs on CPU. Requires the `onnx` package (training environment only).
"""

import argparse
import json
import os
import shutil
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tree_node(cfg, name_in, name_out):
    a = {k: [] for k in ("nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_values", "nodes_modes",
                         "nodes_truenodeids", "nodes_falsenodeids", "target_treeids", "target_nodeids",
                         "target_ids", "target_weights")}
    for t, tr in enumerate(cfg["trees"]):
        for i in range(len(tr["leaf"])):
            leaf = bool(tr["leaf"][i])
            a["nodes_treeids"].append(t)
            a["nodes_nodeids"].append(i)
            a["nodes_featureids"].append(0 if leaf else int(tr["feat"][i]))
            a["nodes_values"].append(0.0 if leaf else float(tr["thr"][i]))
            a["nodes_modes"].append("LEAF" if leaf else "BRANCH_LEQ")      # x <= thr -> left, as scikit-learn
            a["nodes_truenodeids"].append(0 if leaf else int(tr["left"][i]))
            a["nodes_falsenodeids"].append(0 if leaf else int(tr["right"][i]))
            if leaf:
                a["target_treeids"].append(t)
                a["target_nodeids"].append(i)
                a["target_ids"].append(0)
                a["target_weights"].append(float(tr["value"][i]))
    return helper.make_node("TreeEnsembleRegressor", [name_in], [name_out], domain="ai.onnx.ml", n_targets=1,
                            aggregate_function="SUM", base_values=[float(cfg["baseline"])], post_transform="NONE",
                            name="approve_stage_trees", **a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--check", default="", help="dev jsonl: compare graph output with the numpy stage")
    args = ap.parse_args()
    d = args.export_dir
    cfg = json.load(open(os.path.join(d, "approve_stage.json")))
    path = os.path.join(d, "model.onnx")
    if not os.path.exists(path + ".nostage"):
        shutil.copy(path, path + ".nostage")
    m = onnx.load(path + ".nostage")
    g = m.graph
    n_extra = len(cfg["features"]) - 1                           # everything except the network logit
    g.input.append(helper.make_tensor_value_info("stage_x", TensorProto.FLOAT, ["b", n_extra]))
    out = [o for o in g.output if o.name == "approve"][0]
    # rename the network's approve output to approve_net everywhere it is produced
    for node in g.node:
        node.output[:] = ["approve_net" if o == "approve" else o for o in node.output]
    g.output.remove(out)
    g.node.extend([
        helper.make_node("Unsqueeze", ["approve_net", "stage_axes"], ["approve_net_2d"]),
        helper.make_node("Concat", ["approve_net_2d", "stage_x"], ["stage_in"], axis=1),
        tree_node(cfg, "stage_in", "approve_2d"),
        helper.make_node("Squeeze", ["approve_2d", "stage_axes"], ["approve"]),
    ])
    g.initializer.append(helper.make_tensor("stage_axes", TensorProto.INT64, [1], [1]))
    g.output.append(helper.make_tensor_value_info("approve", TensorProto.FLOAT, ["b"]))
    if not any(op.domain == "ai.onnx.ml" for op in m.opset_import):
        m.opset_import.append(helper.make_opsetid("ai.onnx.ml", 3))
    onnx.checker.check_model(m)
    onnx.save(m, path)
    meta_p = os.path.join(d, "meta.json")
    meta = json.load(open(meta_p))
    meta["inputs"] = list(dict.fromkeys(meta.get("inputs", []) + ["stage_x"]))
    meta["stage_in_graph"] = True
    meta["stage_caps"] = cfg["caps"]
    json.dump(meta, open(meta_p, "w"), indent=2)
    print("embedded %d trees into %s (%.1f MB)" % (len(cfg["trees"]), path, os.path.getsize(path) / 1e6))

    if args.check:
        from decide.approve_stage import ApproveStage
        from decide.student_runtime import StudentRuntime
        rt = StudentRuntime(d)
        st = ApproveStage(cfg)
        rows = [json.loads(l) for l in open(args.check) if l.strip()][:300]
        # graph output vs (graph network logit -> numpy stage)
        rt_ns = StudentRuntime(d, onnx_file="model.onnx.nostage")
        diff = [abs(rt.logits("approve", r["state"]) - st.logit(rt_ns.logits("approve", r["state"]), r["state"]))
                for r in rows]
        print("parity on %d dev rows: max |graph - numpy stage| = %.2e" % (len(rows), max(diff)))


if __name__ == "__main__":
    main()
