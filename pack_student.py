"""
Copy an exported student into artifacts/student, splitting the ONNX file into
<100MB chunks (GitHub's per-file limit). The runtime rejoins them in memory.

  python pack_student.py exports/e32 path/to/policy.json [--variant model.onnx]
"""

import argparse
import json
import os
import shutil

CHUNK = 90 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("policy")
    ap.add_argument("--variant", default="model.onnx")
    ap.add_argument("--out", default="artifacts/student")
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    data = open(os.path.join(args.export_dir, args.variant), "rb").read()
    for i in range(0, len(data), CHUNK):
        with open(os.path.join(args.out, "model.onnx.part%02d" % (i // CHUNK)), "wb") as fh:
            fh.write(data[i:i + CHUNK])
    shutil.copy(os.path.join(args.export_dir, "tokenizer.json"), args.out)
    stage = os.path.join(args.export_dir, "approve_stage.json")
    meta = json.load(open(os.path.join(args.export_dir, "meta.json")))
    if os.path.exists(stage) and not meta.get("stage_in_graph"):
        shutil.copy(stage, args.out)   # numpy fallback; not needed once the trees are in the graph
    meta = json.load(open(os.path.join(args.export_dir, "meta.json")))
    meta["variant"] = "model.onnx"
    meta["source_variant"] = args.variant
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    shutil.copy(args.policy, os.path.join(args.out, "policy.json"))
    print("packed", args.out, sorted(os.listdir(args.out)), "%.1f MB" % (len(data) / 1e6))


if __name__ == "__main__":
    main()
