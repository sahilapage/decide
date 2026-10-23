# Model card — Decide

## What it is
One multi-task transformer that answers three typed questions about a customer
support request in a single CPU forward pass: **route** (billing / technical /
account / shipping / general), **approve** (auto-approve vs human review) and
**grounded** (is the drafted reply supported by the attached KB article). Each
request names one decision; the same weights answer all three.

* Backbone: `jhu-clsp/ettin-encoder-32m` (ModernBERT-style, MIT), fully
  fine-tuned. Mean-pooled encoder output feeds three heads (5-way linear,
  MLP over [text, tier × amount], 1-way linear).
* Input: a short typed prompt per decision plus the fields that decision needs;
  `grounded` reads (draft, article) as one sequence pair. Numbers and durations
  are normalised first ("two weeks" → "14 days").
* Serving: ONNX Runtime, fp32, 4 threads, one request at a time. No GPU, no network.
* Size: 32M parameters, 128 MB on disk.

## Training data
The provided labelled split only: 6,500 train tickets, each with all three
labels. Dev (1,500) was never used for fitting weights or calibration; it is
only used to report. No external data, no generated data.

Training: 5 epochs, AdamW, lr 8e-5 (heads 10×), batch 32, cosine schedule,
bf16, one homogeneous batch per decision, all three losses (cross-entropy /
binary cross-entropy — both strictly proper scoring rules) on one encoder.
5-fold cross-fitting gives out-of-fold predictions for calibration.

## Measured quality (dev, scored by `selfcheck.py`)
<!-- CARD-RESULTS:START -->
| | accuracy (all) | quality metric (answered) | ECE | Brier skill | coverage |
|---|---|---|---|---|---|
| route | 0.893 | macro-F1 0.944 | 0.034 | 0.79 | 0.92 |
| approve | 0.893 | cost 0.211 (always-hold 0.382) | 0.029 | 0.66 | 0.84 |
| grounded | 0.997 | macro-F1 1.000 | 0.005 | 0.99 | 0.94 |

p95 latency per call: ~5.5 ms (Apple M5, 4 threads), ~16 ms (i9-13900K, 4 cores pinned).
<!-- CARD-RESULTS:END -->

## Calibration
Temperature (route) and Platt scaling (approve, grounded), fitted on
out-of-fold predictions. Route is deliberately not confident: about 11% of
routing labels do not match what the text supports (vague tickets such as
"can someone help with my account" are labelled almost at random), and the
model's probabilities reflect that rather than claiming 0.99.

## When it abstains
* **route** — when the top queue's probability is low (bottom ~8%): vague or
  generic tickets with no queue-specific content.
* **approve** — where answering hurts the 3:1 cost score most: probabilities
  near the 0.75 break-even, typically amounts just under a tier's cap.
* **grounded** — only when the model is undecided (~6%).
Coverage is kept above the 80% target on every decision.

## Where it fails
* **Approval near the amount caps.** Approval falls off gradually around each
  tier's cap (free ≈ $20–30, pro ≈ $100–120, enterprise ≈ $375–400), and about
  20% of requests well under the cap are still refused for account-history
  reasons the ticket never mentions. Accuracy there is close to a coin flip;
  the model says so (p ≈ 0.5–0.8) instead of guessing confidently.
* **Noisy routing labels.** Errors concentrate on generic tickets and on
  multi-issue tickets where the label does not follow the stated priority.
* **Out-of-distribution text.** Trained on one synthetic-looking help desk;
  new policy wording, new products or other languages will degrade grounding
  and routing. Numbers written in unusual forms ("a couple of days") are not
  normalised.
* The 0.75 approve threshold assumes calibration holds on the deployment
  distribution; if the approval base rate shifts, refit the Platt layer.
