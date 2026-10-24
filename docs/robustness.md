# Robustness and ceilings

This page answers two questions:

1. What happens to Decide when tickets don't look like the training templates?
2. How close is it to the best score this data allows?

All numbers come from `decide/scoring.py`, an exact copy of the `selfcheck.py` maths. Every model is run with its own shipped policy and nothing is refitted on the shifted data, so these are the numbers a judge would see.

## 1. Shifted-dev stress test (`tools/stress_test.py`)

Every dev ticket is rewritten in a way that keeps its label, and the full
three-decision score is computed again. One further variant changes the label
on purpose:

| variant | what changes | in training? |
|---|---|---|
| typos | adjacent-letter swaps in ~6% of words, all four text fields | yes (different typo generator) |
| lower_nopunct | lowercase, punctuation stripped | partly (case only) |
| synonyms | refund→reimbursement, charged→billed, … | no |
| body_shuffle | ticket body sentences reordered | no |
| amount_format | "$136.12" → "136.12 USD" | no |
| kb_shuffle | KB article sentences reordered | yes |
| draft_shuffle | draft sentences reordered | no |
| kb_distractor | one unrelated KB sentence appended | yes |
| num_words | digits ↔ number words | no |
| draft_ticket_ref | draft opens by quoting the order number / charged amount | no (training uses different courtesy wording, no amounts) |
| all_mixed | three of the above at once | – |
| num_flip | one number shared with the KB is changed, so the label flips to *not grounded* | – (checks sensitivity) |

### What we found

The original model (v5, trained on the raw templates) handled **route and approve**
well: they lost at most about 2 points of accuracy on any variant.

**Grounded was brittle.** Its accuracy fell to:

| variant | grounded accuracy |
|---|---|
| typos | 0.81 |
| kb_shuffle | 0.83 |
| kb_distractor | 0.74 |
| draft_ticket_ref | 0.88 |

Almost every error was a *correct* draft called unsupported with p ≈ 0. These
confident errors also wrecked the abstention score. Two causes:

1. **The exact-match number check.** "14 dasy" in the article didn't match "14 days" in the draft, so the check fired.
2. **The encoder had learned the templates.** It expected the KB article in its usual sentence order and length, and drafts with no extra sentences.

### Fixes

* **Number check** (`decide/normalize.py`, `decide/textfmt.py`):
  * one-typo repair for long number and unit words, and for unit words right after a digit
  * typo-tolerant unit matching
  * numbers copied from the ticket (order IDs, the charged amount) no longer count as unsupported claims
  * **It produces identical features and identical normalised text on all 8,000 clean train+dev rows**, so the clean score is untouched.
* **Augmentation, grounded batches only** (`decide/augment.py`, `--augment 2 --augment_tasks grounded`):
  * two copies of each training ticket, with the same label
  * families: typos, case changes, KB reorder, an inserted unrelated KB sentence, polite no-claim sentences around the draft
  * Route and approve were already robust, so they keep training on the clean rows only.
  * Augmenting every decision (v6a) cost about 2 clean points, mostly on route.

### Results (total score / 100)

| variant | v5 (old) | v6a: all-task aug | v6g1: grounded aug ×1 | v6g2: grounded aug ×2 | **v6c2: + courtesy (shipped)** |
|---|---|---|---|---|---|
| clean | 85.22 | 83.36 | 84.47 | 84.72 | **84.66** |
| typos | 69.39 | 79.05 | 80.74 | 80.56 | **81.43** |
| lower_nopunct | 78.85 | 77.56 | 78.50 | 77.39 | 78.80 |
| synonyms | 85.02 | 83.03 | 84.46 | 84.31 | 84.53 |
| body_shuffle | 84.85 | 82.24 | 84.11 | 84.30 | 84.12 |
| amount_format | 85.33 | 83.90 | 84.43 | 84.56 | 84.36 |
| kb_shuffle | 71.19 | 83.36 | 84.37 | 84.72 | **84.66** |
| draft_shuffle | 82.32 | 83.16 | 79.85 | 81.68 | **84.52** |
| kb_distractor | 65.17 | 82.58 | 81.06 | 83.72 | **83.65** |
| num_words | 85.22 | 83.36 | 84.47 | 84.72 | 84.66 |
| draft_ticket_ref | 75.58 | 59.25 | 73.61 | 81.44 | **79.65** |
| all_mixed | 71.48 | 75.20 | 78.09 | 79.62 | **79.82** |
| num_flip (label flips) | 85.06 | 83.32 | 84.36 | 84.65 | 84.63 |
| **mean of shifted variants** | 78.29 | 79.75 | 81.56 | 82.74 | **83.04** |
| **mean of variants never seen in training** | 82.45 | 78.93 | 81.35 | 82.63 | **82.95** |

Columns are seed 0, except v6g2, which is seed 3. Seed 0 of v6g2 dropped to
61 on draft_ticket_ref, and seed 0 of v6a dropped to 59.

**Only the courtesy-trained recipe is stable across seeds** on draft_ticket_ref
(grounded 0.953 and 0.979 for its two seeds).

Raw files are in `docs/stress/`. Re-print the table with
`python tools/compare_stress.py docs/stress/*.json`.

### Round 2: one-sentence articles and new topics (v6t2, v6L2)

Hand-written, judge-style tickets (`tests/judge_cases.json`, 26 cases) exposed one more weakness.

**The weakness.** Every training KB article has 3–4 sentences. When a judge types a one-sentence article, the model doubts correct drafts: a paraphrase got p = 0.02–0.11.

**The fix.** The `kb_trim` family drops KB sentences the draft does not use. It is applied to correct and incorrect drafts alike, so article length gives no hint about the label. For correct drafts it may also cut the draft to one sentence, since a subset of supported claims is still supported.

**A second initialisation.** We also tried starting from `KRLabsOrg/tinylettuce-ettin-17m-en`: the same Ettin-17M weights, already fine-tuned for RAG hallucination detection (MIT licence, published Aug 2025).

| | v6c2 s0 (round 1) | v6t2 s0 | **v6t2 s3 (shipped)** | v6L2 s0 (TinyLettuce init) | v6L2 s3 |
|---|---|---|---|---|---|
| cross-fitted clean dev | 83.8 | 83.0 | **84.1** | 84.3 | 83.6 |
| draft_ticket_ref | 79.7 | 80.3 | **84.6** | 80.2 | 83.3 |
| kb_single (one-sentence KB) | 70.6 | 75.6 | **78.1** | 74.6 | 75.9 |
| mean of variants never seen in training | 83.0 | 82.5 | **83.8** | – | – |
| judge cases | 21 / 26 | 25 / 26 | **25 / 26** | 25 / 26 | 24 / 26 |

The TinyLettuce start was level on clean data and no better on the shifts, so we kept the plain Ettin initialisation.

Fresh-clone `selfcheck.py` for v6t2 s3: **84.98**, all gates PASS, p95 6.3 ms.
With the approve second stage added (model card, "Approve second stage"):
**85.87**, cross-fitted 85.1; robustness unchanged (kb_single 78.8,
draft_ticket_ref 85.4, judge cases 25 / 26).

### Round 3: "complex" queries (longer, multi-part tickets and long articles)

The hidden set is described as containing more complex queries. We added five
label-preserving variants to `tools/stress_test.py` (run with `--only`):

| variant | what changes |
|---|---|
| long_body | 2–4 neutral sentences from tickets with the same route and approve labels |
| multi_issue | a second, less urgent issue from another queue ("Separately, less urgent: …") |
| long_kb | 3–5 sentences from other KB articles inserted into the article (never ones stating the draft's numbers) |
| thread | the ticket wrapped in an email thread: greeting, sign-off, quoted earlier message |
| complex_all | all four at once |

Moved sentences never contain one of the four approve-blocking phrases or a plan
name, so the true approve label can't change. No input exceeds the 192-token
limit, so truncation is not a factor.

**Weak spot: long KB articles.** Every train/dev article has 2–4 sentences. With
7–9 sentences, the encoder misses the supporting sentence and calls correct
drafts unsupported with p ≈ 0, which costs abstention as well as accuracy.

**Fix, at inference only** (`decide/textfmt.py` `focus_kb`): for articles longer
than 5 sentences, the encoder reads the (at most 4) sentences that share the
draft's rarer terms, ranked by summed rarity (`decide/kb_idf.json`, built from
training KB sentences by `tools/build_kb_idf.py`). The number/scope flags still
read the full article. No clean article has more than 4 sentences, so
`format_input` is byte-identical on all 8,000 train+dev rows, and the
fresh-checkout `selfcheck.py` score is unchanged at **85.87**.

Shipped model (approve stage included), same variants with and without the focus:

| variant | without | with focus | grounded acc |
|---|---|---|---|
| clean, kb_distractor, multi_issue, long_body, thread | – | identical | – |
| long_kb | 78.57 | **81.44** | 0.914 → 0.957 |
| complex_all | 76.35 | **80.89** | 0.899 → 0.957 |

Also tried: training with a `kb_long` augmentation family (2–5 distractor
sentences; `train_mt.py --augment_extra kb_long`). Once the focus is on, it
added nothing on long_kb (0.959 vs 0.957), lost a little on kb_single (0.912 vs
0.927), and cross-fitted 83.5 vs 84.1 on clean dev, so it is not shipped.

**Remaining gap: multi-issue routing.** With a second issue from another queue,
route accuracy falls from 0.891 to 0.847 (total 85.7 → 83.7). Approve is
unaffected (0.891 vs 0.899) and grounded is perfect.

### Cost on clean data

Each run's policy is cross-fitted on dev, with 10 halves per run:

| run | cross-fitted dev |
|---|---|
| v5 seeds 0–3 (old recipe) | 82.7 / 83.8 / 83.7 / 84.8, mean **83.8** |
| v6c2 seeds 0 / 3 | 83.8 / 83.2 |

So the new recipe matches the old recipe's average seed on clean data, and it is
much stronger once the text changes. The old shipped seed (84.8) was the best
of four on dev, so part of its lead is selection luck.

Full `selfcheck.py` over HTTP from a fresh clone (Linux, 4 logical CPUs):
**84.66**, all gates PASS, p95 6.3 ms.

## 2. How close to the ceiling? (`tools/ceilings.py`)

These numbers come from 5-fold out-of-fold predictions on train: 6,500 rows, each
predicted by a model that never saw it.

**Route error among the most confident k% of tickets:**

| k | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|
| error | 0.055 | 0.055 | 0.055 | 0.056 | 0.057 | 0.110 |

The curve is **flat at 5.5%**. Even the tickets the model is surest about are
mislabelled 5.5% of the time, which matches the brief's warning about
inconsistent routing labels. No confidence signal can rank those errors.

The best possible route abstention score, given our full error rate, is
1 − 0.0554/0.1095 = **0.494**. Ours is **0.484**.

**Approve** behaves the way the data's structure predicts:

| k | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|
| error | 0.000 | 0.001 | 0.001 | 0.034 | 0.070 | 0.116 |

The confident "hold" cases (over the cap, or one of the disqualifying phrases)
are essentially never wrong. Errors start once the model reaches the requests
that look approvable. About 14–16% of those are refused for account history the
ticket never shows. On dev, requests under the cap with no disqualifying phrase
are approved only 84% of the time.

A transparent rules model (tier, amount, the five disqualifying phrases, gradient
boosting) reaches 0.81 accuracy against the network's 0.885. The network has
learned more of the policy than those phrases.

**Grounded** has an out-of-fold error of 0.05% (3 of 6,500), so it is at its ceiling on in-distribution data.
