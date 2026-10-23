"""
Multi-task fine-tuning of one shared encoder on all three decisions.

  python train_mt.py --model jhu-clsp/ettin-encoder-32m --out runs/e32 --folds 5
  python train_mt.py --model ... --out runs/e32_full --full          # final fit on all train

Per fold it writes the out-of-fold logits (used to fit calibration and, for
teachers, as distillation targets) and dev logits. With --soft, the route,
approve and grounded targets are blended with a teacher's out-of-fold
probabilities (knowledge distillation).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from decide.net import DecideNet
from decide.textfmt import DECISIONS, format_input, tabular

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def encode(tok, rows, max_len):
    """Tokenise every (row, decision) once. Returns dict decision -> list of features."""
    feats = {}
    for d in DECISIONS:
        a, b = zip(*[format_input(d, r["state"]) for r in rows])
        if b[0] is None:
            enc = tok(list(a), truncation=True, max_length=max_len)
        else:
            enc = tok(list(a), list(b), truncation="longest_first", max_length=max_len)
        feats[d] = [{k: enc[k][i] for k in enc.keys()} for i in range(len(rows))]
    tabs = np.stack([tabular(r["state"]) for r in rows])
    return feats, tabs


def collate(tok, items, tabs_rows, device):
    batch = tok.pad(items, return_tensors="pt")
    out = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    out["tab"] = torch.from_numpy(tabs_rows).to(device)
    return out


@torch.no_grad()
def predict(model, tok, feats, tabs, device, bs=128):
    model.eval()
    n = len(tabs)
    res = {"route": np.zeros((n, 5), np.float32), "approve": np.zeros(n, np.float32), "grounded": np.zeros(n, np.float32)}
    for d in DECISIONS:
        order = np.argsort([len(f["input_ids"]) for f in feats[d]])
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            b = collate(tok, [feats[d][i] for i in idx], tabs[idx], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r, a, g = model(**b)
            if d == "route":
                res["route"][idx] = r.float().cpu().numpy()
            elif d == "approve":
                res["approve"][idx] = a.float().cpu().numpy()
            else:
                res["grounded"][idx] = g.float().cpu().numpy()
    return res


def train_one(args, tr_rows, soft, device, eval_sets):
    """soft: dict with teacher probs aligned to tr_rows or None. eval_sets: name -> rows."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = DecideNet(args.model, dropout=args.dropout).to(device)
    feats, tabs = encode(tok, tr_rows, args.max_len)
    y_r = torch.tensor([QUEUES.index(r["labels"]["route"]) for r in tr_rows])
    y_a = torch.tensor([float(r["labels"]["approve"]) for r in tr_rows])
    y_g = torch.tensor([float(r["labels"]["grounded"]) for r in tr_rows])

    # homogeneous batches per decision, interleaved
    def batches():
        out = []
        for d in DECISIONS:
            w = args.task_weight.get(d, 1.0)
            idx = np.random.permutation(len(tr_rows))
            for s in range(0, len(idx), args.bs):
                out.append((d, idx[s:s + args.bs]))
        random.shuffle(out)
        return out

    steps = args.epochs * len(batches())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = {d: 0.0 for d in DECISIONS}
        cnt = {d: 0 for d in DECISIONS}
        for d, idx in batches():
            b = collate(tok, [feats[d][i] for i in idx], tabs[idx], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r, a, g = model(**b)
            if d == "route":
                logp = F.log_softmax(r.float(), -1)
                hard = F.one_hot(y_r[idx], 5).float().to(device)
                if soft is not None:
                    tgt = (1 - args.alpha) * hard + args.alpha * torch.from_numpy(soft["route"][idx]).to(device)
                else:
                    tgt = hard * (1 - args.ls) + args.ls / 5
                loss = -(tgt * logp).sum(-1).mean()
            else:
                z = (a if d == "approve" else g).float()
                yy = (y_a if d == "approve" else y_g)[idx].to(device)
                if soft is not None:
                    yy = (1 - args.alpha) * yy + args.alpha * torch.from_numpy(soft[d][idx]).to(device)
                loss = F.binary_cross_entropy_with_logits(z, yy)
            loss = loss * args.task_weight.get(d, 1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            tot[d] += loss.item()
            cnt[d] += 1
        msg = " ".join("%s %.4f" % (d, tot[d] / max(1, cnt[d])) for d in DECISIONS)
        print("  epoch %d  %s  (%.0fs)" % (ep, msg, time.time() - t0), flush=True)

    preds = {}
    for name, rows in eval_sets.items():
        f, t = encode(tok, rows, args.max_len)
        preds[name] = predict(model, tok, f, t, device)
    return model, tok, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", type=str, default="")
    ap.add_argument("--full", action="store_true", help="train once on all of train, save weights")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ls", type=float, default=0.0, help="label smoothing on route (no teacher)")
    ap.add_argument("--soft", type=str, default="", help="teacher OOF npz for distillation")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tw", type=str, default="", help="task weights e.g. route=1,approve=1,grounded=1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_extra", type=str, default="", help="extra jsonl rows appended to train")
    args = ap.parse_args()
    args.task_weight = {k: float(v) for k, v in (x.split("=") for x in args.tw.split(",") if x)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    device = "cuda"

    tr = load("data/train.jsonl")
    dv = load("data/dev.jsonl")
    soft_all = None
    if args.soft:
        z = np.load(args.soft)
        soft_all = {"route": z["route_prob"].astype(np.float32),
                    "approve": z["approve_prob"].astype(np.float32),
                    "grounded": z["grounded_prob"].astype(np.float32)}

    if args.full:
        model, tok, preds = train_one(args, tr, soft_all, device, {"dev": dv})
        np.savez(os.path.join(args.out, "dev_logits.npz"), **{k: v for k, v in preds["dev"].items()})
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
        tok.save_pretrained(os.path.join(args.out, "tokenizer"))
        model.encoder.config.save_pretrained(os.path.join(args.out, "encoder_config"))
        print("saved full model to", args.out)
        return

    yr = [QUEUES.index(r["labels"]["route"]) for r in tr]
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    only = {int(x) for x in args.only_folds.split(",") if x}
    for f, (a, b) in enumerate(skf.split(tr, yr)):
        if only and f not in only:
            continue
        print("fold %d: train %d  val %d" % (f, len(a), len(b)), flush=True)
        soft = None
        if soft_all is not None:
            soft = {k: v[a] for k, v in soft_all.items()}
        _, _, preds = train_one(args, [tr[i] for i in a], soft, device,
                                {"val": [tr[i] for i in b], "dev": dv})
        np.savez(os.path.join(args.out, "fold%d.npz" % f), val_idx=b,
                 **{"val_" + k: v for k, v in preds["val"].items()},
                 **{"dev_" + k: v for k, v in preds["dev"].items()})
        del preds
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
