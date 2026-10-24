# Methodology: from a 77-point baseline to 85.9

This document records how the shipped Decide model was built: the problem
framing, what we learned from the data, and each stage from the gate-1 baseline
to the final model. Every stage lists what changed, why, how it was measured
and the result. Every number comes from a measured run; the commit times are
from the repository history.

| Stage | What changed | Dev score (selfcheck) | Honest estimate |
|---|---|---|---|
| 0 | Starter stub (uniform probabilities) | ~0 | |
| 1 | Gate-1 baseline: sparse n-grams + three linear heads | 77.0 | |
| 2 | One shared transformer encoder (Ettin-32M), multi-task, ONNX on CPU | 81.95 | |
| 3 | Number normaliser + rule features for grounded | grounded 99.1% → 99.98% (out-of-fold) | |
| 4 | Noise-aware calibration | +0.65 | |
| 5 | Cost-derived decision rule, short typed prompts, per-decision abstention | 84.6 | |
| 6 | Ettin-32M → Ettin-17M (same quality, 2.5× faster); int8 rejected | 85.22 | 84.7 (lucky seed; mean seed 83.8) |
| 7 | Robustness: stress test + grounded-only augmentation | 84.98 | 84.1 |
| 8 | Approve second stage (60 boosted trees, compiled into the graph) | **85.87** | **85.1** |
| 9 | Long-KB focus for complex queries (inference only) | 85.87 (unchanged) | long-KB stress +2.9 |

"Honest estimate" is the cross-fitted dev score: the policy is fitted on one
half of dev and scored on the other, over 10 random halves. The in-sample
selfcheck number is slightly optimistic, because dev was used to fit the
policy's scales and to choose between candidates.

---

## 1. Problem and evaluation protocol

**Task.** For each customer-support ticket (`subject`, `body`, `customer_tier`,
`amount`, `draft_reply`, `kb_article`), answer one typed question per call:

* **route:** one of 5 queues, with probabilities
* **approve:** yes/no + probability; a wrong approval costs 3× a wrong hold
* **grounded:** yes/no + probability that the draft is supported by the article

Each answer can also abstain.

**Constraints.**
* One shared model for all three decisions.
* CPU only at inference, p95 < 100 ms (full marks at ≤ 25 ms).
* No network calls while serving, no generated data, artifacts < 500 MB.

**Metric** (`selfcheck.py`, unchanged from the starter):
* **quality 40%:** answered-set macro-F1 / cost against the best trivial
  policy, scaled down below 80% coverage
* **calibration 25%:** half ECE, half Brier skill
* **latency 20%**
* **abstention 15%:** `1 − risk@80% / full error`, ranked by the reported
  confidence

There are three pass/fail gates: schema, latency and reachability.

**Data and how we used it.**
* 6,500 labelled train tickets and 1,500 dev tickets, each with all three
  labels. The hidden test set is the held-back ~20% of the same dataset.
* **Network weights never see dev.** Dev is used only to fit a handful of
  calibration scales and abstain thresholds, and to compare candidates.
* Every claim is reported both in-sample and cross-fitted.

**Honest predictions for fitting.** Calibration and the approve second stage
are fitted on **out-of-fold** (OOF) predictions. A 5-fold
`StratifiedKFold` (by route, `random_state=0`) gives every training ticket a
prediction from a model that never saw it. Fitting on in-sample predictions
would learn the network's over-confidence on its own training data.

---

## 2. What the data told us (before modelling)

1. **Route labels are partly random.**
   * About 5.5% of route labels don't match the ticket text.
   * The OOF error among the model's most confident 10/20/40/60/80% of tickets
     is 5.5 / 5.5 / 5.5 / 5.6 / 5.7%: the curve is flat.
   * So no confidence signal can find those errors. The best possible route
     abstention score is ≈ 0.494.
2. **Approve depends on a soft cap plus hidden history.**
   * Each tier has an amount cap (free ≈ $20–30, pro ≈ $100–120,
     enterprise ≈ $375–400).
   * The approval rate falls *gradually* through the cap zone: for free,
     67% at $15–20, 51% at $20–25, 22% at $25–27.5. Near the cap the label is
     close to a coin flip.
   * Four phrases always block approval ("dispute a charge", "outside your
     usual policy", "raised something similar", "signed up recently").
   * Even well under the cap, ~15% of requests are refused for account history
     the ticket never mentions. An n-gram model scores AUC 0.46 on those
     refusals and the network 0.52: they are unpredictable.
3. **Grounded errors are one-slot changes.**
   * A wrong draft differs from its article in exactly one slot: a number
     ("15" vs "30 minutes"), an audience ("administrators" vs "all users") or
     a scope ("monthly" vs "annual").
   * After normalising numbers and units, **no correct draft in train or dev
     states a number + unit that its article doesn't contain.**
4. **Together these set a ceiling of roughly 87**, so
   beyond a point the remaining gains come from calibration, abstention and
   robustness, not raw accuracy.

---

## 3. Stage 1: gate-1 baseline (77.0)

**Goal:** a scored, schema-valid submission within the first two hours, and a
measure of how hard the problem is.

**Model** (`decide/baseline.py`, `training/train_baseline.py`):
* **Shared backbone:** one sparse vocabulary. A binary `CountVectorizer`,
  1–3-grams, `min_df=2`, fitted once over all normalised text fields.
* **Three logistic-regression heads**, all on features from that one vocabulary:
  * route: n-grams of subject + body
  * approve: the same n-grams + tier one-hot + tier × amount-bucket one-hot
  * grounded: phrases in the draft but **not** in the article, plus phrases
    they share
* **Calibration and abstention:** a first version of the policy.

**Result:** 77.0 on dev. That established that the problem is learnable, and
that the gap to close lay mostly in route and grounded.

---

## 4. Stage 2: one shared transformer, served on CPU (81.95)

**Change:**
* Replaced the sparse backbone with a pretrained encoder,
  **Ettin-32M** (`jhu-clsp`, ModernBERT recipe, MIT licence).
* Fine-tuned it **multi-task** on all three decisions (`training/train_mt.py`).
* Exported it as one ONNX graph with three outputs, served by ONNX Runtime on
  CPU (`training/export_onnx.py`, `decide/student_runtime.py`).

**How one model serves three questions** (`decide/textfmt.py`, `decide/net.py`):
* Each call builds a **typed prompt**: `route: <subject> | <body>`,
  `approve: <tier> $<amount> | <subject> | <body>`, or
  `grounded: <draft>` paired with `<kb_article>`.
* The shared encoder reads it, its output is mean-pooled, and three small
  heads read the result:
  * route: `Linear(h, 5)`
  * approve: an MLP over `[h, aux]`
  * grounded: an MLP over `[h, aux]`
* More than 99% of the parameters are shared.
* **Grounded is a cross-encoder.** Draft and article go through the encoder
  *together*, so attention can compare them word by word. Embedding them
  separately would blur a one-word difference.

**Training setup:**
* Batches are single-decision and interleaved at random, so every step
  updates the shared encoder from one task.
* AdamW (weight decay 0.01); heads train at 10× the encoder learning rate.
* Cosine schedule with 6% warm-up, gradient clipping at 1.0, bf16 autocast on
  GPU.

**Result:** 81.95.

---

## 5. Stage 3: number normalisation and rule features for grounded

**Change** (`decide/normalize.py`, `decide/textfmt.py`, `decide/net.py`):
* **Normaliser:**
  * number words, including hundreds and thousands
  * every duration in one unit, using the data's own conventions: month = 30
    days, year = 365, week = 7, hour = 60 minutes, quarter hour = 15 minutes
  * business days kept as they are
  * applied to every decision
* **Rule features** in a 71-dimensional auxiliary vector:
  * tier one-hot, tier × amount bucket, log amount
  * for grounded: "the draft states a number fact the article doesn't", how
    many, the share supported, and "scope broadened" ("all plans" vs "all
    paid plans")
* **Learned rule weights.** The flag and scope features are added straight to
  the grounded logit through weights that **start at −8** and stay trainable.
  The network starts out knowing the rule, but can override it.

**Why:** neural encoders are unreliable at unit arithmetic ("a month" vs
"sixty days"), and before this most confident grounded errors were exactly
that. A deterministic check after normalisation is exact.

**Result:** grounded out-of-fold accuracy 99.1% → **99.98%** (3 errors in
6,500).

---

## 6. Stage 4: noise-aware calibration (+0.65)

**Problem:** temperature scaling assumes a perfect model could reach 100%
confidence. With 5.5% random route labels, even the clearest ticket is right
only ~95% of the time.

**Change** (`decide/policy.py`):
* **Route:** `p = (1 − ε)·softmax(z/T + bias) + ε/5`. The per-queue bias and
  the noise floor ε model the label noise.
* **Approve and grounded:** bounded Platt scaling with a learned floor and
  ceiling, `lo + (hi − lo)·sigmoid(a·z + b)`.
  * The ceiling encodes the hidden-history refusals.
  * The bounds (`|a| ≤ 1.5`, `|b| ≤ 5`) stop the near-perfect grounded head
    from producing exact 0 or 1 probabilities.
  * A plain Newton Platt fit diverged on grounded, so it was replaced with
    bounded L-BFGS.
* **What is fitted where:**
  * structural parameters (ε, bias, floor, ceiling): the 6,500 OOF predictions
  * scales only (T, a, b): the shipped model's dev predictions

**Result:** calibration part 0.832 → 0.858, **+0.65** total.

---

## 7. Stage 5: decision rule, typed prompts and abstention (84.6)

**Decision rule for approve.** Calibrated `p` with a 3:1 cost means approving
has expected cost `3(1 − p)` and holding has expected cost `p`. So approve
only when **p > 0.75**. Grounded uses p > 0.5, and route takes the top queue.

**Short typed prompts.** Replacing long prompts ("Subject: … Message: …") with
short tags (`route:`) cut the token count with no loss in quality.

**Abstention, tuned per decision** against an in-process copy of the judges'
scorer (`decide/scoring.py`, built from `selfcheck.py`'s own functions):
* We search abstain shares up to 15% (coverage floor 0.85, a margin above the
  80% target) and keep the smallest share within 0.002 of the best quality.
* For approve, three rules are searched (by cost, by confidence, and a
  cost-ratio rule).
* **Shipped behaviour:**
  * route abstains on the ~15% lowest top-probability tickets (the vague ones)
  * approve abstains where p is between 0.17 and 0.83 (~10% of requests)
  * grounded essentially never abstains

**Result:** 84.6.

---

## 8. Stage 6: model size and latency (85.22)

**Measurements** (`tools/bench_cpu.py`). p95 is the mean over the three
decisions, in-process, with the CPU pinned to 2 physical cores:

| Model | p95 | Cross-fitted dev |
|---|---|---|
| Ettin-32M fp32 | 14.2 ms | 83.4 |
| Ettin-32M int8 (4 quantisation variants) | 7.8–8.0 ms | 80.0–80.4 |
| **Ettin-17M fp32** | **5.4 ms** | same quality as 32M |

**Decisions:**
* **int8 rejected.** It costs ~3 points, almost all on grounded (accuracy
  99.1% → 92.0%): comparing exact numbers needs precise activations.
* **The smaller fp32 model was chosen instead.** It is as fast as int8 with
  no quality loss.
* Bigger encoders (150M, 400M) were no better, so the limit is the labels,
  not model capacity.
* **Serving settings:** ONNX Runtime with 4 threads and spinning on, a Rust
  tokenizer, and weights stored in < 100 MB chunks that are joined in memory
  at startup.

**Result:** 85.22 in-sample, 84.7 cross-fitted, p95 2.5–6.3 ms depending on
the machine.

**A caution we recorded.** Seed 3 was the best of four 17M seeds. The four
seeds score 82.7 / 83.8 / 83.7 / 84.8 cross-fitted (mean 83.8), so part of
85.22 is selection luck.

---

## 9. Stage 7: robustness to shifted text (84.98, stress mean 78.3 → 83.8)

**Stress test** (`tools/stress_test.py`):
* Dev tickets are rewritten in **label-preserving** ways: typos,
  lowercase/no punctuation, synonyms, reordered ticket/KB/draft sentences, an
  unrelated KB sentence, amount formats, digits ↔ number words, drafts quoting
  the order number, one-sentence articles, and three shifts at once.
* One **label-flipping** check: a number changed, so the draft becomes
  unsupported.
* Each variant is scored with the shipped policy and no refit.

**Finding:**
* Route and approve lost at most ~2 points of accuracy on any variant.
* **Grounded was brittle:** typos 0.81, KB reordered 0.83, one extra KB
  sentence 0.74. Nearly every error was a correct draft called unsupported at
  p ≈ 0, which also hurt abstention.

**Changes:**
1. **Number check made robust** (no retraining):
   * one-typo repair for long number/unit words
   * typo-tolerant unit matching
   * numbers copied from the ticket (order ids, the charged amount) don't
     count as claims
   * features are identical on all 8,000 clean rows
2. **Grounded-only augmentation** (`decide/augment.py`,
   `--augment 2 --augment_tasks grounded`): two extra copies of each training
   ticket, each with 1–3 of: typos, case changes, reordered KB sentences, an
   inserted unrelated KB sentence (never one stating a draft number), polite
   no-claim openers/closers, and trimming KB sentences the draft doesn't use.
   * Labels never change. This is augmentation, not data generation.
   * Augmenting all three decisions cost ~2 clean points (mostly route), so
     only grounded batches see the copies.
3. **Candidate selection:** v6a / v6g1 / v6g2 / v6c2 / v6t2, plus a
   hallucination-detector initialisation (TinyLettuce) that was no better.
   Candidates were compared on three things together: cross-fitted clean
   score, stress mean, and hand-written judge-style cases (`tests/`).

**Result (shipped v6t2, seed 3):**
* 84.98 in-sample, 84.1 cross-fitted
* mean over held-out shift types 83.8 (original model 82.5)
* worst stress case 78 (original 65)
* judge cases 25/26

---

## 10. Stage 8: approve second stage (85.87, cross-fitted 85.1)

**Problem:** the network sees the amount only through coarse tier × amount
buckets, and it misses part of the soft-cap policy.

**Change** (`decide/approve_stage.py`, `tools/fit_approve_stage.py`):
* A 60-tree gradient-boosted correction over
  `[network approve logit, tier, amount ÷ tier cap, amount == 0, disqualifying-phrase flags]`.
* Caps are estimated from the training labels (the amount where approval falls
  to 50%).
* It is fitted on **out-of-fold** logits, so it corrects the network's real
  over-confidence (the network memorises hidden-history refusals in training).
* It is compiled into the same ONNX graph as a `TreeEnsembleRegressor` node
  (`tools/embed_stage.py`).
  * It cannot run without the shared encoder's logit, and it leaves route and
    grounded untouched, so it is still one model.
  * It adds ~0.1 ms.
* Putting the same features into the network instead did not help.

**Result:**
* approve OOF accuracy 88.5% → 90.3%, Brier 0.094 → 0.076
* **+1.0** cross-fitted total (84.1 → 85.1)
* fresh-clone selfcheck **85.87**

---

## 11. Stage 9: complex queries (inference-only fix)

**Test sets** (`tools/stress_test.py --only ...`):
* longer ticket bodies
* multi-issue tickets
* email threads
* **long KB articles** (3–5 unrelated sentences inserted, 7–9 sentences in
  total)
* all of these combined

Moved sentences never contain an approve-blocking phrase or a plan name, so
labels stay true. No input exceeds the 192-token limit, so truncation is not a
factor.

**Finding:** only long KB articles hurt. Grounded fell 1.00 → 0.91, again as
confident "unsupported" calls on correct drafts.

**Change** (`decide/textfmt.py` `focus_kb`):
* For articles longer than 5 sentences (longer than anything in training), the
  encoder reads the ≤ 4 sentences that share the draft's rarer terms.
* Terms are 5-letter stems plus numbers, weighted by rarity across training KB
  sentences (`decide/kb_idf.json`).
* The number and scope flags still read the full article.
* Clean inputs are byte-identical on all 8,000 rows, so selfcheck is unchanged
  at 85.87.

**Result:**
* long KB 78.6 → **81.4**
* everything combined 76.4 → **80.9**
* clean, thread, long body, multi-issue and one-extra-sentence variants
  unchanged

**Also tried:** a `kb_long` training augmentation. On top of the focus it
added nothing (grounded 0.959 vs 0.957) and cost ~0.6 cross-fitted, so it was
not shipped.

---

## 12. What did not help (ablations)

Every experiment below used the same seed and was measured on the same 6,500
OOF predictions or the same dev halves:

| Idea | Result |
|---|---|
| Bigger encoders (32M, 150M, 400M) | same quality, slower |
| Distillation from large teachers (Ettin-150M/400M, DeBERTa-v3-large, a Laya-initialised encoder) | no gain |
| Seed ensembles (logit averaging of 4 seeds) | 84.0 vs 84.8 for the best single seed |
| Model soups | hurt grounded |
| int8 quantisation | −3 points |
| **Loss and architecture variants** on Ettin-17M (OOF total; base 84.43) | |
| L2-SP anchor to pretrained weights | 84.32 |
| layer-wise learning-rate decay (0.85) | 84.09 |
| R-Drop consistency | 84.05 |
| attention pooling (per-decision) | 83.12 |
| focal loss (γ = 2) | 81.97 |
| GCE noise-robust loss (q = 0.7) | 80.68 |
| Coverage floor 0.80 instead of 0.85 | 84.78 vs 84.72 (noise) |

**Reading.**
* Route and approve accuracy stayed within ±0.3 across all variants. The
  variants that lost did so on **grounded confidence**: a few confident
  grounded errors push grounded abstention from 1.0 to 0.4–0.8.
* Losses that up-weight hard examples (focal) or flatten confidence (GCE)
  hurt, because the hard examples here are mostly mislabelled.
* All of this confirms the ceiling analysis.
* A second run of the same losses on the shipped recipe (grounded
  augmentation and the approve tree stage) found GCE
  slightly *positive* (+0.24, about 1σ) and GCE + layer-wise LR decay +0.29.
  So the sign of GCE depends on the setup; in neither setup is the gain large
  enough to ship.

---

## 13. Extensibility and data scaling

**Adding a fourth decision**:
* Leave-one-out: train on route + approve only, then add grounded as if it
  were new (one prompt, one head).
* Fine-tuning for **89 s** on top of the trained model, replaying 30% of the
  old decisions' batches, gives **83.2 vs 83.6** for joint training (cross-fit
  noise ±1.1). Route and approve stay within 1 point.
* Head-only on a frozen encoder keeps the old answers bit-identical, but
  reaches only 83% on grounded: comparing two texts is a skill the encoder has
  to learn.

**Training-set size:** the full shipped pipeline retrained on 5,000 vs 6,500
train tickets scores 84.26 vs **85.05** cross-fitted (+0.79, one seed each).
The 6,500 run independently reproduces the shipped model's 85.1.

---

## 14. Verification

* **Fresh-clone test:** `git clone`, `./start.sh`, then `selfcheck.py` on a
  clean CPU-only Linux box. 85.87, 4,500/4,500 valid responses, all gates PASS.
  Online install (Python 3.10) and offline wheel install (Python 3.11) were
  both verified.
* **Independent reproduction** on a second machine and GPU: 85.05 cross-fitted
  from scratch.
* **`test.py`:** in-process run over any dataset. It prints every answer, the
  timing and the accuracy, and reproduces selfcheck's 85.87 by sending calls in
  the same shuffled order.
* **Hand-written cases:** `tests/judge_cases.json` (25/33, including 7 known
  gaps from an independent audit) and `tests/complex_cases.json` (29/31).
* **Starter integrity:** `app.py` and `selfcheck.py` are byte-identical to the
  starter. `model.py` keeps the starter's class and method signatures.

---

## 15. Final configuration

| Component | Setting |
|---|---|
| Encoder | `jhu-clsp/ettin-encoder-17m` (MIT), mean pooling |
| Heads | route `Linear(h, 5)`; approve and grounded MLP(`[h, aux71]` → 256 → 1); grounded flag weights init −8 |
| Approve second stage | 60 boosted trees, compiled into the ONNX graph |
| Prompts | short typed prompts; grounded as a (draft, article) pair |
| Max length | 192 tokens |
| Optimiser | AdamW, lr 1e-4 (heads 1e-3), weight decay 0.01, cosine schedule, 6% warm-up, grad clip 1.0 |
| Training | 6 epochs, batch 32, dropout 0.1, bf16 on GPU, head init seed 1234, seed 3 |
| Augmentation | 2 label-preserving copies per ticket, grounded batches only |
| Out-of-fold | 5-fold `StratifiedKFold` by route, `random_state=0` |
| Calibration | route: temperature + per-queue bias + noise floor; binaries: bounded Platt with floor/ceiling |
| Decisions | approve p > 0.75; grounded p > 0.5; route argmax |
| Abstention | coverage ≥ 0.85; route ~15% lowest top-probability; approve p ∈ (0.17, 0.83) |
| Serving | ONNX Runtime fp32, 4 threads, Rust tokenizer, 68 MB, p95 ~5–6 ms on 2 cores |

Commands to reproduce are in the README ("Reproduce").

---

## 16. Limitations

* **Route** is capped near 89–90% accuracy by random labels; vague tickets are
  abstained on by design.
* **Paraphrased disqualifying phrases** (a reworded chargeback threat or
  exception request) can still be approved at p ≈ 0.85. Generalised features
  exist (tab layout v3) but need retraining.
* **Multi-issue tickets** cost route ~4 points of accuracy.
* **Cross-unit durations** ("24 hours" vs "one day") are not matched by the
  number check.
* Dev guided the seed, the recipe and the policy scales, so **~85** is the
  honest expectation on the hidden set, not 85.87.

---

## 17. Team

* **Sahil:** models and training recipe, data analysis, calibration and
  abstention policy, robustness augmentation, approve second stage, ceiling
  analysis, stress and judge-case suites, serving hardening.
* **Arhan:** CPU latency work and benchmarks (layouts, threads, int8 study),
  independent verification and reproduction, loss and architecture ablations,
  the extensibility experiment, the complex-query test sets and long-KB fix,
  the training-size study, `test.py`, the demo page, repository organisation
  and figures.
