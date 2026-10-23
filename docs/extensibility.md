# Adding a fourth decision: measured, not assumed

The brief says the decision types share a backbone *"so adding a fourth one later
is cheap"*. The dataset has no fourth label, and generating one is out of scope.
So we test the claim on a decision we *can* score: we **pretend grounded is the
new decision**.

1. Train the shared model on **route + approve only** (`base`). It has never
   seen a grounded question.
2. Add grounded afterwards, the way a fourth decision would be added: one new
   prompt string (`grounded:`) and one new head, trained **on top of `base`**,
   with no retraining from scratch.
3. Compare with the model trained on all three together (`joint`), and check
   whether route and approve changed.

Grounded is the hardest of the three to add: it compares two texts, a skill the
other two never teach the encoder. Adding an easier decision (another label on
the same ticket) should be cheaper still.

## Results

Ettin-17M, the shipped recipe (v6t2: `--augment 2 --augment_tasks grounded`),
seed 3. Each model is scored on dev exactly as `selfcheck.py` does, with the
policy cross-fitted over 10 random dev halves (± is the spread). RTX 3060
wall-clock.

| model | dev total | route acc | approve acc | grounded acc | train time |
|---|---|---|---|---|---|
| joint: all three together, from scratch | **83.61 ± 1.12** | 0.889 | 0.882 | 0.997 | 160 s |
| base: route + approve only | 72.21 ± 1.54 | 0.881 | 0.871 | – | 75 s |
| **+ grounded, fine-tune with replay, all labels** | **83.19 ± 1.21** | 0.881 | 0.889 | **0.999** | **89 s** |
| + grounded, stronger fine-tune (lr 1e-4, replay 0.5) | 83.33 ± 1.09 | 0.884 | 0.879 | 0.999 | 134 s |
| + grounded, 1,000 labels only | 75.04 ± 1.39 | 0.881 | 0.872 | 0.883 | 59 s |
| + grounded, frozen encoder (new head only) | 72.90 ± 1.37 | *bit-identical* | *bit-identical* | 0.827 | 77 s |

## What this shows

* **Adding a decision later costs about 90 seconds on top of the trained model,
  and it lands where joint training lands** (83.2 against 83.6, inside the
  ±1.1 noise).
* **The existing decisions keep their accuracy.** Route and approve move by
  under 1 point either way. Replay (a share of old-decision batches mixed into
  the new decision's training) is what keeps them in place.
* **Head-only (frozen encoder) keeps the old answers bit-for-bit identical, but
  cannot teach a new skill.** Grounded stays at 83%, carried mostly by the
  number-mismatch features. For a decision that only re-reads the same ticket
  (urgency, sentiment, a new queue), frozen mode is the zero-risk option.
  A decision needing a new skill gets the short fine-tune.
* **Labels matter more than training time.** 1,000 grounded labels reach 88%;
  all 6,500 reach 99.9%. A text-comparison decision needs thousands of
  examples. A simpler one needs fewer.

## How to add decision #4 (e.g. `urgent`)

1. **Prompt**: add `"urgent": "urgent:"` to `PROMPT_SETS` in
   `decide/textfmt.py`, and a branch in `format_input` choosing which fields it
   reads.
2. **Head**: add `self.urgent_head` in `decide/net.py` (binary: an MLP on the
   pooled vector; choice: a `Linear(h, n_classes)`), and a fourth ONNX output
   in `export_onnx.py`.
3. **Train on top of the shipped model**:
   ```bash
   python train_mt.py --model jhu-clsp/ettin-encoder-17m --prompt short --full --seed 3 \
       --tasks urgent --init_from <shipped run>/model.pt --replay 0.3 --epochs 4 --lr 3e-5
   ```
   (add `--freeze_encoder` instead of `--replay` to keep the other answers
   bit-identical)
4. **Calibrate and serve**: the binary Platt / abstain code in `decide/policy.py`
   is generic. Fit it on the new decision's dev logits, and the existing
   `binary` response type in the API covers it.

## Reproduce

```bash
C="--model jhu-clsp/ettin-encoder-17m --prompt short --augment 2 --augment_tasks grounded --full --seed 3"
python train_mt.py $C --out runs/ext/joint --epochs 6 --lr 1e-4
python train_mt.py $C --out runs/ext/base  --epochs 6 --lr 1e-4 --tasks route,approve
A="--tasks grounded --init_from runs/ext/base/model.pt"
python train_mt.py $C --out runs/ext/tune_n0   $A --replay 0.3 --epochs 4 --lr 3e-5
python train_mt.py $C --out runs/ext/frozen_n0 $A --freeze_encoder --epochs 10 --lr 1e-4
python tools/extend_eval.py --new grounded --base runs/ext/base --ref runs/ext/joint runs/ext/tune_n0 runs/ext/frozen_n0
```
