# Model card — Decide

## What it is
One multi-task transformer that answers three typed questions about a customer
support request in a single CPU forward pass: **route** (billing / technical /
account / shipping / general), **approve** (auto-approve vs human review) and
**grounded** (is the drafted reply supported by the attached KB article). Each
request names one decision; the same weights answer all three.

* Backbone: `jhu-clsp/ettin-encoder-17m` (ModernBERT recipe, MIT), fully
  fine-tuned. Mean-pooled encoder output feeds three heads: route (linear, 5
  logits), approve (MLP over [text, aux], then a 60-tree boosted layer over
  that logit, compiled into the same ONNX graph), grounded (MLP over
  [text, aux] plus two learned flag weights).
* Input: a short typed prompt (`route:` / `approve:` / `grounded:`) plus the
  fields that decision needs; `grounded` reads (draft, article) as one sequence
  pair. Numbers and durations are normalised first ("two weeks" → "14 days",
  "a month" → "30 days", "a quarter of an hour" → "15 minutes").
* aux vector: tier one-hot, tier × amount bucket, log amount; and for
  grounding: "draft states a number+unit or range the article does not",
  how many, share supported, and "draft drops a qualifier from an *all …*
  scope" ("all plans" vs "all paid plans").
* Serving: one ONNX graph (encoder, three heads and the approve tree layer),
  ONNX Runtime fp32, 4 threads, one request at a time. No GPU, no network.
* Size: 17M parameters, 68 MB.

## Training data
The provided labelled split only: 6,500 train tickets with all three labels.
No external data, no LLM-generated data and no new labels. Dev (1,500) is
never used to fit network weights or the approve tree layer (fitted on train
out-of-fold logits). It is used to (a) refit three calibration scales and the
abstain thresholds for the shipped network, (b) pick one of two training seeds,
and (c) compare candidate recipes (`docs/report.md`) and sanity-check feature
rules. So the in-sample dev score is optimistic; the cross-fitted estimate
corrects only (a), and ~84.5–85 is the honest expectation on unseen data.

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
| approve | 0.899 | cost 0.201 (always-hold 0.405) | 0.017 | 0.68 | 0.90 |
| grounded | 1.000 | macro-F1 1.000 | 0.000 | 1.00 | 1.00 |

Total 85.9 / 100 (quality 0.856, calibration 0.876, latency 1.00, abstention
0.647), measured over HTTP from a fresh clone. Cross-fitted dev estimate (policy
fitted on one half, scored on the other, 10 halves): **85.1** (range 82.8–86.7).
Seeds of the same recipe differ by ~1 point on dev and this seed was the
better of two, so treat ~84.5–85 as the expectation on unseen data.

p95 latency per call: 6.3 ms over HTTP on 4 logical CPUs (two physical cores of
an i9-13900K, the shape of a 4-vCPU VM); about 2.5 ms on an Apple M5.

Robustness (`tools/stress_test.py`, `docs/robustness.md`): the same dev tickets
rewritten with typos, reordered or extra KB sentences, case changes, number
words, ticket references in the draft, one-sentence KB articles, etc. Worst
variant 78.8 (the un-augmented model fell to 65.2). Hand-written judge-style
tickets: `tests/judge_cases.json` 25 / 33 (the 7 cases added from an
independent audit — paraphrased refusal reasons, non-numeric slot changes —
are the known gaps below) and `tests/complex_cases.json` 29 / 31 (multi-issue
tickets, quoted threads, mixed language, multi-claim drafts).
<!-- CARD-RESULTS:END -->

## Approve second stage (inside the ONNX graph)
The shared network sees the amount only through coarse tier × amount buckets.
A 60-tree gradient-boosted layer (`decide/approve_stage.py`; compiled into
`model.onnx` as a `TreeEnsembleRegressor` node by `tools/embed_stage.py`,
parity 6e-7 with the numpy version) takes the
network's approve logit plus amount ÷ tier cap (caps estimated from the
labels: the amount where approval falls to 50%), amount == 0 and five
disqualifying-phrase flags mined from the training labels. It is fitted on
out-of-fold network logits, so it corrects the network's real errors:
approve accuracy 88.5% → 90.3% and Brier 0.094 → 0.076 out of fold, total +1.0
(6,500 OOF rows, paired sd 0.28; +1.0 on dev cross-fitted too). It adds well
under 0.1 ms and changes nothing for route or grounded. Putting the same
features into the network instead did not help: the network memorises the
training labels, and only a layer fitted on out-of-fold logits can learn to
correct that. Below the cap with no disqualifying phrase, ~15% of requests are
still refused and nothing in the text predicts which (n-gram AUC 0.46, network
0.52): that is the hidden account history, and approve probabilities top out
near 0.85–0.90 to say so (calibrated ceiling 0.90).

## Calibration
Label-noise aware, fitted on out-of-fold predictions:
* route: `p = (1-eps)·softmax(z/T + bias) + eps/5`. About 5-6% of routing
  labels are unrelated to the ticket text, so even clear tickets are right only
  ~95% of the time; temperature alone cannot express that plateau.
* binaries: Platt scaling with a learned floor/ceiling
  (`lo + (hi-lo)·sigmoid(a·z+b)`), with the slope bounded so the fit stays
  stable on the nearly separable grounded decision (its probabilities still
  reach ~0.0000 / 0.99999 — correctly, given 0 errors on dev).
* The noise parameters come from 6,500 out-of-fold rows; only T, a, b are
  refitted on the shipped network's dev predictions.

## When it abstains
* **route** — lowest top-queue probability (~15% of dev): generic tickets with
  no queue-specific content ("can someone help with my account please").
* **approve** — the least certain ~10%: p between 0.17 and 0.83 (the
  confidence rule won the tuning in `decide/policy.py`), typically amounts near
  a tier's cap.
* **grounded** — essentially never; it is not unsure.
Coverage is capped at 85% minimum so the 80% target holds on unseen data.
Approve commits to "yes" only when p > 0.75, where 3·(1−p) < p.

## Where it fails
* **Approval near the amount caps.** Approval falls off gradually around each
  tier's cap (free ≈ $20–30, pro ≈ $100–120, enterprise ≈ $375–400) and ~15%
  of requests well under the cap are refused for account-history reasons the
  ticket never shows. Accuracy there is near a coin flip; the model says so.
* **Round dollar amounts the training text only saw refused.** 99% of training
  amounts have cents (47 of 5,228 are whole dollars), and the only training
  tickets that mention "$140" or "$200" in their text were refused. The model
  has picked that up from the amount it reads in the prompt: an enterprise
  refund of $200.00–$200.99 gets p ≈ 0.05–0.10 (hold) while $199.99 or $205 get
  p ≈ 0.85; $100 and $140 are pulled down to the abstain zone. Rare in data
  that looks like the training set; visible in `docs/figures/15_approve_curve.png`.
* **Paraphrased refusal reasons.** The disqualifying phrases are learned from
  the training templates; a hand-written paraphrase ("I'll file a chargeback
  with my card company") is not always recognised and can be auto-approved
  at p ≈ 0.85 (`tests/judge_cases.json` a7–a11).
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
  one-sentence KB articles (grounded 93%). Not handled: a duration stated in
  another unit ("one day" vs "24 hours") — matching across units was tried and
  rejected because it un-flagged 13 unsupported train/dev drafts.
* **Very long inputs** are clipped to 4,000 characters per field before
  tokenising (longest field in train/dev: 482).
* **int8 is not safe here.** Dynamic int8 quantisation halves latency but drops
  grounded accuracy from 99% to 92% (exact number comparison needs precise
  activations), so we ship fp32.
