"""
Export a trained DecideNet to one ONNX graph for CPU serving.

  python training/export_onnx.py runs/e32_full artifacts/student [--policy runs/e32_policy.json]

Writes model.onnx (fp32), model.int8.onnx (dynamic int8), tokenizer.json,
meta.json and policy.json into the output folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json
import os
import shutil

import torch

from decide.net import DecideNet
from decide.textfmt import set_tab_version, tab_dim


class Wrap(torch.nn.Module):
    def __init__(self, net, use_tt):
        super().__init__()
        self.net = net
        self.use_tt = use_tt

    def forward(self, input_ids, attention_mask, tab, token_type_ids=None):
        r, a, g = self.net(input_ids, attention_mask, tab, token_type_ids if self.use_tt else None)
        return torch.log_softmax(r, -1), a, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--policy", default="")
    ap.add_argument("--max_len", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_args = json.load(open(os.path.join(args.run, "args.json")))
    set_tab_version(run_args.get("tab_version", 1))
    net = DecideNet(os.path.join(args.run, "encoder_config"), from_config=True,
                    pool=run_args.get("pool", "mean"))
    sd = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    use_tt = net.encoder.config.model_type in ("deberta-v2", "bert")
    if hasattr(net.encoder.config, "_attn_implementation"):
        net.encoder.config._attn_implementation = "eager"
    w = Wrap(net, use_tt).eval()

    ids = torch.ones(1, 48, dtype=torch.long)
    mask = torch.ones(1, 48, dtype=torch.long)
    tab = torch.zeros(1, tab_dim())
    inputs = (ids, mask, tab) + ((torch.zeros(1, 48, dtype=torch.long),) if use_tt else ())
    names = ["input_ids", "attention_mask", "tab"] + (["token_type_ids"] if use_tt else [])
    dyn = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "tab": {0: "b"},
           "route": {0: "b"}, "approve": {0: "b"}, "grounded": {0: "b"}}
    if use_tt:
        dyn["token_type_ids"] = {0: "b", 1: "s"}
    fp32 = os.path.join(args.out, "model.onnx")
    torch.onnx.export(w, inputs, fp32, input_names=names, output_names=["route", "approve", "grounded"],
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    print("exported", fp32, os.path.getsize(fp32) / 1e6, "MB")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = os.path.join(args.out, "model.int8.onnx")
    quantize_dynamic(fp32, q, weight_type=QuantType.QInt8)
    print("quantized", q, os.path.getsize(q) / 1e6, "MB")

    tok_dir = os.path.join(args.run, "tokenizer")
    shutil.copy(os.path.join(tok_dir, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        if os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(tok_dir, f), os.path.join(args.out, f))
    meta = {"max_len": args.max_len or run_args["max_len"], "use_token_type_ids": use_tt,
            "encoder": run_args["model"], "inputs": names,
            "prompt_style": run_args.get("prompt", "long"), "tab_version": run_args.get("tab_version", 1)}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    if args.policy:
        shutil.copy(args.policy, os.path.join(args.out, "policy.json"))
    print("done", meta)


if __name__ == "__main__":
    main()
