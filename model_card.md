# Model card — Decide

## What it is
One multi-task transformer that answers three typed questions about a customer
support request in a single CPU forward pass: **route** (billing / technical /
account / shipping / general), **approve** (auto-approve vs human review) and
**grounded** (is the drafted reply supported by the attached KB article). Each
request names one decision; the same weights answer all three.

* Backbone: `jhu-clsp/ettin-encoder-17m` (ModernBERT recipe, MIT), fully
  fine-tuned. Mean-pooled encoder output feeds three heads: route (linear, 5
  logits), approve (MLP over [text, aux]), grounded (MLP over [text, aux] plus
  two learned flag weights).
* Input: a short typed prompt (`route:` / `approve:` / `grounded:`) plus the
  fields that decision needs; `grounded` reads (draft, article) as one sequence
  pair. Numbers and durations are normalised first ("two weeks" → "14 days",
  "a month" → "30 days", "a quarter of an hour" → "15 minutes").
* aux vector: tier one-hot, tier × amount bucket, log amount; and for
  grounding: "draft states a number+unit or range the article does not",
  how many, share supported, and "draft drops a qualifier from an *all …*
  scope" ("all plans" vs "all paid plans").
* Serving: ONNX Runtime fp32, 4 threads, one request at a time. No GPU, no network.
* Size: 17M parameters, 68 MB.

## Training data
The provided labelled split only: 6,500 train tickets with all three labels.
No external data, no LLM-generated data and no new labels. Dev (1,500) is
never used to fit network weights. It is used to (a) refit three calibration
scales and the abstain thresholds for the shipped network and (b) pick one of
two training seeds — so the in-sample dev score below is slightly optimistic;
the cross-fitted estimate is the honest one.

Training: 6 epochs, AdamW, lr 1e-4 (heads 10×), batch 32, cosine schedule,
bf16, one homogeneous batch per decision, cross-entropy / binary
cross-entropy (both strictly proper scoring rules) on the shared encoder.
5-fold cross-fitting gives out-of-fold predictions for calibration.

Augmentation (grounded batches only, `decide/augment.py`): two rewritten copies
of each training ticket with the *same* label — typos, case changes, KB
sentences reordered, one unrelated KB sentence inserted, polite no-claim
sentences around the draft, KB sentences the draft does not use dropped (so
one-sentence articles are familiar; correct drafts may be cut to one sentence,
which keeps them supported). These are surface rewrites of the provided rows,
not new data; they stop the encoder from relying on the templates' exact
spelling, sentence order and length.

## Measured quality (dev, `selfcheck.py` against the live service)
<!-- CARD-RESULTS:START -->
| | accuracy (all) | quality metric (answered) | ECE | Brier skill | coverage |
|---|---|---|---|---|---|
| route | 0.891 | macro-F1 0.946 | 0.024 | 0.78 | 0.85 |
| approve | 0.890 | cost 0.204 (always-hold 0.368) | 0.013 | 0.65 | 0.85 |
| grounded | 1.000 | macro-F1 1.000 | 0.000 | 1.00 | 1.00 |

Total 85.0 / 100 (quality 0.842, calibration 0.875, latency 1.00, abstention
0.630), measured from a fresh clone. Cross-fitted dev estimate (policy fitted
on one half, scored on the other, 10 halves): **84.1** (range 81.6–85.5).
Seeds of the same recipe differ by ~1 point on dev and this seed was the
better of two, so treat ~83.5–84 as the expectation on unseen data.

p95 latency per call: 6.3 ms over HTTP on 4 logical CPUs (two physical cores of
an i9-13900K, the shape of a 4-vCPU VM); about 2.5 ms on an Apple M5.

Robustness (`tools/stress_test.py`, `docs/robustness.md`): the same dev tickets
rewritten with typos, reordered or extra KB sentences, case changes, number
words, ticket references in the draft, one-sentence KB articles, etc. Mean over
shifts never used in training: 83.8 (un-augmented model 82.5); worst variant
78.1 (was 65.2). Hand-written judge-style tickets (`tests/judge_cases.json`):
25 / 26 as expected; the miss ("Do you have an office in Berlin?") abstains.
<!-- CARD-RESULTS:END -->

## Calibration
Label-noise aware, fitted on out-of-fold predictions:
* route: `p = (1-eps)·softmax(z/T + bias) + eps/5`. About 5-6% of routing
  labels are unrelated to the ticket text, so even clear tickets are right only
  ~95% of the time; temperature alone cannot express that plateau.
* binaries: Platt scaling with a learned floor/ceiling
  (`lo + (hi-lo)·sigmoid(a·z+b)`), bounded so a nearly separable decision
  (grounded) cannot drive probabilities to exactly 0 or 1.
* The noise parameters come from 6,500 out-of-fold rows; only T, a, b are
  refitted on the shipped network's dev predictions.

## When it abstains
* **route** — lowest top-queue probability (~11%): generic tickets with no
  queue-specific content ("can someone help with my account please").
* **approve** — up to 15%, where answering hurts the 3:1 cost score most:
  probabilities near the 0.75 break-even, typically amounts just under a tier's cap.
* **grounded** — essentially never; it is not unsure.
Coverage is capped at 85% minimum so the 80% target holds on unseen data.
Approve commits to "yes" only when p > 0.75, where 3·(1−p) < p.

## Where it fails
* **Approval near the amount caps.** Approval falls off gradually around each
  tier's cap (free ≈ $20–30, pro ≈ $100–120, enterprise ≈ $375–400) and ~20%
  of requests well under the cap are refused for account-history reasons the
  ticket never shows. Accuracy there is near a coin flip; the model says so.
* **Noisy routing labels.** Confident routing "errors" are mostly mislabels
  ("Package not delivered" labelled billing). They cap accuracy near 90% for
  any model; our probabilities top out around 0.95 rather than 0.99. Measured
  out-of-fold: the most confident 10% of tickets are wrong 5.5% of the time,
  the same as the most confident 80% (5.7%) — a flat floor no confidence signal
  can rank, so route abstention sits within 0.01 of its ceiling
  (`tools/ceilings.py`, `docs/ceilings.json`).
* **Grounding outside the templates.** The number and scope checks assume the
  conventions of this help desk (a month = 30 days). They tolerate one-letter
  typos and ignore numbers copied from the ticket (order IDs, the charged
  amount), but new phrasings ("a couple of days"), new policy types or other
  languages fall back to the encoder alone. Hardest stress variant left:
  one-sentence KB articles (grounded 93%).
* **int8 is not safe here.** Dynamic int8 quantisation halves latency but drops
  grounded accuracy from 99% to 92% (exact number comparison needs precise
  activations), so we ship fp32.
