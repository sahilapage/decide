# Decide: strategy, experiments and audit

This is the team's consolidated research and engineering record. Every number
below was measured by us and can be re-run with the tool named next to it.

| Score | Value |
|---|---|
| Judges' scorer (`selfcheck.py`), fresh clone, dev | **85.87 / 100**, all gates PASS |
| Cross-fitted estimate (policy fitted on one half of dev, scored on the other) | 85.1 |
| Best any model can reach on this data (`docs/ceiling.md`) | about 87 |

## 1. The rubric and the strategy

| Axis | Weight | Measured as | What wins it |
|---|---|---|---|
| Quality | 40% | Route and grounded macro-F1 on answered cases; approve cost at 3:1 scaled so that always-hold = 0.25. Each decision is scaled by coverage/0.8 | Accuracy where the labels allow it, the 0.75 approve threshold, and abstaining where it pays |
| Calibration | 25% | ½ (1 − ECE/0.2) + ½ Brier skill | Probabilities that are honest (low ECE) and informative (Brier skill); label noise must show up in them |
| Latency | 20% | p95 per call, one at a time; 25 ms or less = full marks, 100 ms = 0 | A small encoder on ONNX Runtime with a fast tokenizer |
| Abstention | 15% | 1 − risk@80% ÷ full error, ranked by reported confidence. The abstain flag is ignored | Confidence that ranks your own errors |

Pass/fail gates:
* valid schema on every response
* p95 < 100 ms
* a model card
* one shared model
* CPU only, no network while serving
* no hard-coded answers

The strategy, in order:
1. Get latency to full marks with a tiny encoder.
2. Shape each question to what it compares:
   * grounded is a cross-encoder over the (draft, article) pair
   * approve gets the tier × amount
   * route is single-text
3. Make the probabilities reflect the label noise.
4. Measure how close each decision is to its ceiling, and spend effort only where there is room.
5. Harden against surface shifts, in case the hidden set or a judge's typed example differs.

## 2. The frontier research

Full survey: `docs/research/frontier_survey.md` (about 5,800 words, cited).
Source notes: `docs/research/notes/`. The parts that changed what we built:

| Area | Strongest relevant work (verified source) | What we took |
|---|---|---|
| "System One" decision models | TypeSafe **Jev** (hosted, 236–405 ms per call in third-party tests). Convai **Laya**, a 421M ModernBERT-large: its HF card reports ECE 0.47 before a temperature refit and an escalate head with AUROC 0.30 | Same contract (state + typed question → structured answer + probability), 25× smaller, and with abstention that works |
| Small encoders | **Ettin** suite (arXiv 2507.11412), ModernBERT recipe, 17M–1B. 17M = GLUE 79.2, 32M = 83.5 | Ettin-17M: on our data 17M ≈ 32M ≈ 150M in quality, at 2.7× the speed of 32M |
| Grounding detection | MiniCheck (arXiv 2404.10774: synthetic hard negatives matter, −14.8 balanced accuracy without them), AlignScore, LettuceDetect / TinyLettuce (Ettin-17M, RAGTruth F1 68.5) | A cross-encoder pair, symbolic number checks, label-preserving hard augmentation. TinyLettuce init tested; no gain |
| Calibration | Guo et al. 2017; Kull et al. 2019 (Dirichlet); calibration under label noise | Noise-aware calibrators: an ε-floor softmax for route, Platt with floor and ceiling for the binaries |
| Selective prediction | Geifman & El-Yaniv; Feng et al. ICLR 2023 (selection heads rarely beat MSP); Jaeger et al. 2023 | Plain calibrated confidence. A measured floor shows route is already at its ceiling |
| Multi-task optimisation | Kurin et al. 2022, Xin et al. 2022: tuned scalarisation matches PCGrad / GradNorm | Equal-weighted multi-task loss |
| Noisy labels | GCE (Zhang & Sabuncu 2018), R-Drop, L2-SP | Tested: GCE +0.24 (about 1σ); R-Drop −1.44 |
| Quantisation | Bondarenko et al. 2021, 2023 (outlier dimensions break int8) | Explains our −3 points with int8; we ship fp32 |
| Test-time adaptation, RL | TENT is unreliable on text and hurts calibration. With full labels, RL reduces to cost-sensitive classification | Skipped, with reasons |

## 3. Candidate approaches (decision matrix)

| Candidate | Expected gain | Effort | Latency | Risk | Result |
|---|---|---|---|---|---|
| Sparse n-gram multi-head baseline | reference | low | <1 ms | low | 77.0 (gate 1) |
| Fine-tuned shared encoder + typed prompts + task-shaped inputs | large | medium | 5–6 ms | low | **chosen**: 84–85 |
| Larger encoder (32M–400M) or distillation from one | small (labels are noisy) | medium | 2–10× | low | no gain |
| Seed ensembles / weight soups | small | low | ×n | soups broke grounded | no gain |
| int8 quantisation | latency only (already full marks) | low | 0.5× | accuracy | −3 points, rejected |
| Robustness augmentation (grounded only) | protects the hidden set and typed demos | medium | 0 | small clean cost | **chosen**: worst stress variant 65 → 79, clean unchanged |
| Approve second stage (trees on the network logit + amount ÷ cap + phrases) | +0.5–1.5 | low | +0.1 ms | is it "one model"? | **chosen**: +1.0 cross-fitted, +1.0 out of fold |
| Noisy-label losses (GCE, R-Drop, LLRD, attention pooling, L2-SP) | small | low | 0 | low | GCE +0.24 (1σ, confirming), others ≈0 or negative |
| JAX/XLA port | none (latency is full marks) | high | ≈ | high | not pursued |

## 4. Experiment tracker

Metric: the selfcheck total. "Cross-fit" means dev, with the policy fitted on half and scored on the other half, 10 repeats. "OOF" means the same on the 6,500 out-of-fold train rows, which has less noise.

| # | Experiment | Baseline | New | Notes / conclusion |
|---|---|---|---|---|
| 1 | Sparse n-gram baseline | – | 77.0 dev | gate 1 |
| 2 | Ettin-32M multi-task, typed prompts | 77.0 | 81.95 → 84.6 dev | the core recipe |
| 3 | Number/duration normaliser + "unsupported number" flags | grounded OOF 99.1% | 99.98% | the cleanest single win on grounded |
| 4 | Noise-aware calibration (ε-floor; Platt with floor and ceiling) | plain temperature | +0.65 OOF | calibration part 0.832 → 0.858 |
| 5 | Dynamic int8 (4 variants) | 83.4 cross-fit | 79.9–80.4 | grounded 99% → 92%; rejected |
| 6 | Teachers (Ettin 150M/400M, DeBERTa-v3-L, Laya) + distillation | – | no gain | the limit is label noise, not capacity |
| 7 | Weight soups | – | grounded 96.2% | rejected |
| 8 | Ettin-17M vs 32M | same quality | 13.4 → 5.4 ms (2 physical cores) | shipped 17M |
| 9 | 4-seed logit ensemble | 84.78 best seed, 83.77 mean seed | 84.01 | no gain (`docs/research/ensemble_eval.txt`) |
| 10 | Typo-tolerant number check, ticket-number exclusion | stress "typos" 65.9 | 69.4 | features identical on all 8,000 clean rows |
| 11 | Augmentation on all tasks (v6a) | 83.8 mean seed | 82.7 / 82.3 cross-fit | robust, but costs clean route |
| 12 | Grounded-only augmentation ×1 / ×2 (v6g1/2) | 83.8 | 83.7–84.0 | unstable on ticket-ref drafts |
| 13 | + courtesy family (v6c2) | 83.8 | 83.8 / 83.2 | selfcheck 84.66; stress shift mean 78.3 → 83.0 |
| 14 | + kb_trim family (v6t2) | 83.8 | 83.0 / **84.1** | selfcheck 84.98; judge cases 21 → 25/26 |
| 15 | TinyLettuce initialisation (v6L2) | v6t2 | 84.3 / 83.6 cross-fit | no robustness gain; not shipped |
| 16 | Coverage floor 0.80–0.90 | 0.85 | within 0.05 for 0.80–0.85; 0.90 is −0.30 | keep 0.85 |
| 17 | Approve: rules only (tier, amount, phrases) | network 88.5% acc | 81% | the network learns more than the rules |
| 18 | **Approve second stage** | 84.10 cross-fit | **85.13** | +1.0 OOF (paired sd 0.28); selfcheck **85.87** |
| 19 | Approve stage + route probs / request-type words | stage | +0.00 | saturated |
| 20 | Are the hidden refusals predictable? | – | AUC 0.46 (n-grams), 0.52 (network) | no: irreducible |
| 21 | Loss options, OOF with stage (E1 GCE, E2 R-Drop, E3 LLRD, E4 attention pooling, E5 L2-SP) | 84.47 | +0.24 / −1.44 / +0.17 / +0.06 / +0.04 | only GCE looks positive (about 1σ) |
| 22 | GCE + LLRD (E6); full seeds of E1/E6 | – | *running* | ship only on a clear OOF + fresh-clone win |
| 23 | Add a 4th decision later (Arhan, `docs/extensibility.md`) | joint 83.6 | 83.2 in 90 s | sharing is cheap in practice |
| 24 | ECE floor simulation | – | route 0.018, approve 0.014 | ours (0.024 / 0.017) is at the sampling floor |

## 5. Error analysis

* **Route** (89% accuracy):
  * Errors are flat at 5.5% across confidence: the most confident 10% err as often as the most confident 80%.
  * These are the random routing labels the brief warns about.
  * Abstention 0.497 sits against a ceiling of 0.494.
  * The remaining errors are vague tickets ("can someone help with my account"). These get near-flat probabilities and abstain.
* **Approve** (89.9%):
  * Confident holds (over the cap, or a disqualifying phrase) are almost never wrong.
  * Below the cap with no phrase, 15% are refused for hidden reasons that nothing predicts.
  * Near the cap, approval falls gradually (88% → 58% → 30% → 0% across 0.5–1.5× the cap). The second stage fixed most of the network's mistakes there, where it had been 64% accurate.
* **Grounded** (100% on dev):
  * The failure modes are off-template inputs, found with the stress test and the hand-written cases. They are fixed by augmentation; the worst remaining variant is a one-sentence article (93%).
  * `docs/robustness.md` has 13 shifted variants and 26 hand-written tickets.

## 6. How to reproduce

```bash
./start.sh                                                     # serve
python selfcheck.py --url http://localhost:8000/v1/decide      # 85.87
python tools/stress_test.py                                    # robustness table
python tools/judge_cases.py                                    # 26 hand-written cases
```

Training and calibration: see the README, "Reproduce".

## 7. Remaining weaknesses

* **The dev number is slightly optimistic.**
  * Dev picks between two seeds and fits the calibration scales.
  * Seeds of the same recipe differ by about 1 point.
  * Expect about 84.5–85 on unseen data; that is why we quote the cross-fitted 85.1.
* **The approve stage is a second learned component on top of the shared encoder.**
  * It reads only the shared network's approve logit plus ticket metadata, and it leaves route and grounded alone.
  * We describe it as part of the approve head, but a strict judge could call it a second model.
* **Grounded rules assume this help desk's conventions** (a month = 30 days). New policy types or languages fall back to the encoder alone.
* **The hidden-history refusals and random route labels are not learnable.** No change to the model will move those.
