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
import contextlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from decide.net import DecideNet
from decide.textfmt import DECISIONS, format_input, set_prompt_style, tabular

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def amp(device):
    """bf16 autocast on GPU; plain fp32 on CPU (most CPUs have no fast bf16)."""
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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
            with amp(device):
                r, a, g = model(**b)
            if d == "route":
                res["route"][idx] = r.float().cpu().numpy()
            elif d == "approve":
                res["approve"][idx] = a.float().cpu().numpy()
            else:
                res["grounded"][idx] = g.float().cpu().numpy()
    return res


def two_class(z):
    """Binary logit -> 2-class log-probs, so binaries share the route KL code."""
    return F.log_softmax(torch.stack([torch.zeros_like(z), z], -1), -1)


def sym_kl(logp, logq):
    """Symmetric KL between two batches of log-prob rows (consistency / R-Drop)."""
    p, q = logp.exp(), logq.exp()
    return 0.5 * ((p * (logp - logq)).sum(-1) + (q * (logq - logp)).sum(-1)).mean()


def gce(logp, tgt, q):
    """Generalised cross-entropy (Zhang & Sabuncu 2018): (1 - p^q)/q, robust to flipped labels.
    q -> 0 recovers cross-entropy, q = 1 is MAE. Soft targets weight each class."""
    return (tgt * (1 - logp.exp().clamp_min(1e-6) ** q) / q).sum(-1).mean()


def train_one(args, tr_rows, soft, device, eval_sets):
    """soft: dict with teacher probs aligned to tr_rows or None. eval_sets: name -> rows."""
    # Heads always start from the same weights (fixed seed), so runs that differ
    # only in --seed (data order, dropout) stay in one basin and can be
    # weight-averaged into a soup.
    torch.manual_seed(args.init_seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = DecideNet(args.model, dropout=args.dropout, pool=args.pool).to(device)
    if args.init_from:
        # extending a trained model: start from its weights (encoder + existing heads)
        model.load_state_dict(torch.load(args.init_from, map_location=device))
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    n_orig = len(tr_rows)
    aug_tasks = set(args.augment_tasks.split(",")) if args.augment_tasks else set(DECISIONS)
    if args.augment > 0:
        # augmented copies of the TRAINING rows only (fold validation stays clean)
        from decide.augment import augment
        tr_rows = list(tr_rows) + augment(tr_rows, args.augment, seed=args.seed)
        if soft is not None:
            soft = {k: np.concatenate([v] * (1 + args.augment)) for k, v in soft.items()}
        print("  augmented train rows:", len(tr_rows), flush=True)
    feats, tabs = encode(tok, tr_rows, args.max_len)
    y_r = torch.tensor([QUEUES.index(r["labels"]["route"]) for r in tr_rows])
    y_a = torch.tensor([float(r["labels"]["approve"]) for r in tr_rows])
    y_g = torch.tensor([float(r["labels"]["grounded"]) for r in tr_rows])

    # --tasks: decisions trained in this run. --n_labels: label budget for them (a fixed
    # random subset of the original rows, plus those rows' augmented copies).
    # --replay: the other decisions keep that fraction of their batches, so an unfrozen
    # encoder does not forget them while a new decision is added.
    tasks = [d for d in DECISIONS if d in args.tasks.split(",")]
    keep = np.arange(n_orig)
    if args.n_labels:
        keep = np.sort(np.random.RandomState(args.seed).permutation(n_orig)[:args.n_labels])
    copies = 1 + (args.augment if args.augment > 0 else 0)

    # --cons pairs every original row with one of its augmented copies (or, without
    # augmentation, with itself under a second dropout mask = R-Drop), so batches
    # are drawn from the originals only and the copy is looked up per step
    paired = args.cons > 0

    def partner(d, idx):
        if args.augment > 0 and d in aug_tasks:
            return idx + n_orig * np.random.randint(1, args.augment + 1, size=len(idx))
        return idx

    # homogeneous batches per decision, interleaved
    def batches():
        out = []
        for d in DECISIONS:
            if d in tasks:
                # augmented rows only feed the decisions listed in --augment_tasks
                # (with --cons they are looked up per step as partners instead)
                pool = (np.concatenate([keep + c * n_orig for c in range(copies)])
                        if d in aug_tasks and not paired else keep)
            elif args.replay > 0:
                pool = np.random.permutation(n_orig)[:int(args.replay * n_orig)]
            else:
                continue
            idx = np.random.permutation(pool)
            for s in range(0, len(idx), args.bs):
                out.append((d, idx[s:s + args.bs]))
        random.shuffle(out)
        return out

    steps = args.epochs * len(batches())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    enc_groups = [{"params": enc_params, "lr": args.lr}]
    if args.llrd < 1.0:
        # layer-wise lr decay: top layer gets lr, each layer below x llrd, embeddings lowest
        n_layers = model.encoder.config.num_hidden_layers
        by_lr = {}
        for n, p in model.named_parameters():
            if not n.startswith("encoder.") or not p.requires_grad:
                continue
            m = re.search(r"\.layers?\.(\d+)\.", n)
            depth = n_layers - 1 - int(m.group(1)) if m else (n_layers if "embed" in n else 0)
            by_lr.setdefault(args.lr * args.llrd ** depth, []).append(p)
        enc_groups = [{"params": ps, "lr": lr} for lr, ps in by_lr.items()]
    opt = torch.optim.AdamW(enc_groups + [{"params": head_params, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    # L2-SP anchor: pull encoder weights back toward the pretrained start point
    anchor = [p.detach().clone() for p in enc_params] if args.l2sp > 0 else None

    def task_loss(d, out, idx):
        """Returns (loss, log-probs) for decision d on rows idx."""
        r, a, g = out
        if d == "route":
            logp = F.log_softmax(r.float(), -1)
            hard = F.one_hot(y_r[idx], 5).float().to(device)
            if soft is not None:
                tgt = (1 - args.alpha) * hard + args.alpha * torch.from_numpy(soft["route"][idx]).to(device)
            else:
                tgt = hard * (1 - args.ls) + args.ls / 5
        else:
            z = (a if d == "approve" else g).float()
            yy = (y_a if d == "approve" else y_g)[idx].to(device)
            if soft is not None:
                yy = (1 - args.alpha) * yy + args.alpha * torch.from_numpy(soft[d][idx]).to(device)
            logp = two_class(z)
            tgt = torch.stack([1 - yy, yy], -1)
        # grounded labels are clean (100% on dev), so the noise-robust loss skips it
        if args.gce > 0 and d != "grounded":
            return gce(logp, tgt, args.gce), logp
        if args.focal > 0 and d != "grounded":
            # focal loss (Lin et al. 2017), no alpha/pos_weight so probabilities stay unbiased
            return -(tgt * (1 - logp.exp()) ** args.focal * logp).sum(-1).mean(), logp
        return -(tgt * logp).sum(-1).mean(), logp

    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()        # frozen features: no dropout noise
        tot = {d: 0.0 for d in DECISIONS}
        cnt = {d: 0 for d in DECISIONS}
        for d, idx in batches():
            b = collate(tok, [feats[d][i] for i in idx], tabs[idx], device)
            with amp(device):
                out = model(**b)
            loss, logp = task_loss(d, out, idx)
            if paired:
                idx2 = partner(d, idx)
                b2 = collate(tok, [feats[d][i] for i in idx2], tabs[idx2], device)
                with amp(device):
                    out2 = model(**b2)
                loss2, logp2 = task_loss(d, out2, idx2)
                loss = 0.5 * (loss + loss2) + args.cons * sym_kl(logp, logp2)
            loss = loss * args.task_weight.get(d, 1.0)
            if anchor is not None:
                loss = loss + args.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(enc_params, anchor))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if ema is not None:
                # warm EMA: short memory early, args.ema later
                dec = min(args.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(dec).add_(v.float(), alpha=1 - dec)
                        else:
                            ema[k].copy_(v)
            tot[d] += loss.item()
            cnt[d] += 1
        msg = " ".join("%s %.4f" % (d, tot[d] / max(1, cnt[d])) for d in DECISIONS)
        print("  epoch %d  %s  (%.0fs)" % (ep, msg, time.time() - t0), flush=True)

    if ema is not None:
        sd = model.state_dict()
        model.load_state_dict({k: ema[k].to(sd[k].dtype) for k in sd})
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
    ap.add_argument("--init_seed", type=int, default=1234, help="seed for head initialisation")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights, e.g. 0.998")
    ap.add_argument("--prompt", default="long", choices=["long", "short"])
    ap.add_argument("--train_extra", type=str, default="", help="extra jsonl rows appended to train")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--augment", type=int, default=0,
                    help="label-preserving augmented copies per training row (decide/augment.py)")
    ap.add_argument("--cons", type=float, default=0.0,
                    help="consistency weight: symmetric KL between a row and its augmented copy "
                         "(R-Drop on the same row when --augment 0), e.g. 1.0")
    ap.add_argument("--l2sp", type=float, default=0.0,
                    help="L2-SP anchor weight toward the pretrained encoder, e.g. 1e-4")
    ap.add_argument("--gce", type=float, default=0.0,
                    help="generalised cross-entropy q for route/approve (noisy labels), e.g. 0.7; 0 = CE")
    ap.add_argument("--focal", type=float, default=0.0,
                    help="focal loss gamma for route/approve, e.g. 2.0; 0 = CE")
    ap.add_argument("--llrd", type=float, default=1.0,
                    help="layer-wise lr decay per encoder layer below the top, e.g. 0.85; 1 = off")
    ap.add_argument("--pool", default="mean", choices=["mean", "attn"],
                    help="attn = learned per-decision attention pooling")
    ap.add_argument("--augment_tasks", default="", help="decisions that see augmented rows, e.g. grounded (default all)")
    # extending the model with a new decision (see docs/extensibility.md)
    ap.add_argument("--tasks", default="route,approve,grounded", help="decisions trained in this run")
    ap.add_argument("--init_from", default="", help="model.pt of a trained DecideNet to start from")
    ap.add_argument("--freeze_encoder", action="store_true", help="train heads only; other decisions stay bit-identical")
    ap.add_argument("--n_labels", type=int, default=0, help="label budget for --tasks (0 = all train rows)")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="fraction of batches kept for decisions not in --tasks (unfrozen encoder)")
    args = ap.parse_args()
    set_prompt_style(args.prompt)
    args.task_weight = {k: float(v) for k, v in (x.split("=") for x in args.tw.split(",") if x)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

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
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
