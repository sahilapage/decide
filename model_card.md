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
No external or generated data. Dev (1,500) is never used to fit network
weights. It is used to (a) refit three calibration scales and the abstain
thresholds for the shipped network and (b) pick one of four training seeds —
so the in-sample dev score below is slightly optimistic; the cross-fitted
estimate is the honest one.

Training: 6 epochs, AdamW, lr 1e-4 (heads 10×), batch 32, cosine schedule,
bf16, one homogeneous batch per decision, cross-entropy / binary
cross-entropy (both strictly proper scoring rules) on the shared encoder.
5-fold cross-fitting gives out-of-fold predictions for calibration.

## Measured quality (dev, `selfcheck.py` against the live service)
<!-- CARD-RESULTS:START -->
| | accuracy (all) | quality metric (answered) | ECE | Brier skill | coverage |
|---|---|---|---|---|---|
| route | 0.883 | macro-F1 0.943 | 0.035 | 0.78 | 0.89 |
| approve | 0.885 | cost 0.191 (always-hold 0.374) | 0.033 | 0.64 | 0.85 |
| grounded | 1.000 | macro-F1 1.000 | 0.000 | 1.00 | 1.00 |

Total 85.2 / 100 (quality 0.852, calibration 0.846, latency 1.00, abstention
0.667). Cross-fitted dev estimate (policy fitted on one half, scored on the
other, 10 halves): **84.7** (range 82.6–86.2). Out-of-fold on train (6,500,
5-fold): grounded accuracy 99.98%, route 89.2%, approve 89.1%.

p95 latency per call: 2.5 ms over HTTP on an Apple M5; 5.4 ms in-process on
two physical cores of an i9-13900K (the shape of a 4-vCPU VM).
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
  any model; our probabilities top out around 0.95 rather than 0.99.
* **Grounding outside the templates.** The number and scope checks assume the
  conventions of this help desk (a month = 30 days). New phrasings ("a couple
  of days"), new policy types or other languages fall back to the encoder alone.
* **int8 is not safe here.** Dynamic int8 quantisation halves latency but drops
  grounded accuracy from 99% to 92% (exact number comparison needs precise
  activations), so we ship fp32.
