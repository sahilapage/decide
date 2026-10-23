"""
Uniform model soup: average the weights of several full runs fine-tuned from
the same pretrained encoder (different seeds), then predict dev with the
averaged network. Same size and latency as one model.

  python soup.py runs/soup_e32 runs/e32s_full runs/e32s_s1 runs/e32s_s2 runs/e32s_s3
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from decide.net import DecideNet
from decide.textfmt import set_prompt_style
from train_mt import encode, load, predict


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    args0 = json.load(open(os.path.join(runs[0], "args.json")))
    set_prompt_style(args0.get("prompt", "long"))
    sds = [torch.load(os.path.join(r, "model.pt"), map_location="cpu") for r in runs]
    soup = {}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point:
            soup[k] = sum(sd[k].float() for sd in sds) / len(sds)
        else:
            soup[k] = sds[0][k]
    net = DecideNet(os.path.join(runs[0], "encoder_config"), from_config=True)
    net.load_state_dict(soup)
    net = net.cuda()
    tok = AutoTokenizer.from_pretrained(os.path.join(runs[0], "tokenizer"))
    dv = load("data/dev.jsonl")
    f, t = encode(tok, dv, args0["max_len"])
    preds = predict(net, tok, f, t, "cuda")
    np.savez(os.path.join(out, "dev_logits.npz"), **preds)
    torch.save(net.state_dict(), os.path.join(out, "model.pt"))
    for d in ("tokenizer", "encoder_config"):
        if os.path.exists(os.path.join(out, d)):
            shutil.rmtree(os.path.join(out, d))
        shutil.copytree(os.path.join(runs[0], d), os.path.join(out, d))
    a = dict(args0, soup_of=runs)
    json.dump(a, open(os.path.join(out, "args.json"), "w"), indent=2)
    print("soup of", len(runs), "->", out)


if __name__ == "__main__":
    main()
