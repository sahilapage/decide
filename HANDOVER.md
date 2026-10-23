# HANDOVER — start here (for the next Claude session / account)

Date: 2026-10-24. Repo: https://github.com/sahilapage/decide (`main`).

## 1. Read these first, in order

1. `context.md`: the full log.
   * Sections 1–14 are Sahil's side: task, data findings, architecture, results, failed experiments, gotchas, file map, judge Q&A.
   * Section 15 is Arhan's side: independent verification, the second GPU box, research verdicts on test-time training and RL.
2. `README.md` and `model_card.md`: the submission-facing docs.
3. `docs/benchmarks.md`: latency numbers, and why int8 is not shipped.
4. `research_notes/Decide frontier research survey/*.md`: the frontier research survey (section 4 below).

## 2. Current state (all verified)

* **Shipped model:** `artifacts/student/`, Ettin-17M multi-task, recipe `v6c2` seed 0, fp32 ONNX, 68 MB.
  * This recipe augments the grounded batches.
  * The previous model is tagged `v5-e17-s3`.
* **Score:**
  * dev `selfcheck.py` over HTTP from a fresh clone = **84.66 / 100**
  * cross-fitted (honest) estimate = **83.8**
  * route acc 88.9%, approve acc 89.3% (cost 0.204 vs always-hold 0.360), grounded 100%
* **Latency (p95):** 6.3 ms over HTTP on 2 physical cores (4 logical), about 2.5 ms on a Mac (M5).
* **Robustness:** stress-test mean 83.0 (old model 78.3). See `docs/robustness.md`.
* **Teammate branch:** `perf/cpu-latency` is merged.

Quick check from a fresh clone:

```bash
./start.sh &                         # serves :8000
python selfcheck.py --url http://localhost:8000/v1/decide   # expect ~84.7
```

The dataset (`data/*.jsonl`) is **not in git**. It is gitignored because it was handed to participants. Copy it from:
* the starter folder: `~/Downloads/Final_Hackathon_PS_Decide/starter`, or
* the GPU box: `~/sahil/decide/data`

## 3. Machines

| Box | Access | Where things are |
|---|---|---|
| RTX 4090 (Sahil) | `ssh sra@10.27.100.101` | `~/sahil/decide`: venv built with `--system-site-packages` on `~/miniconda3/envs/trackeverything` (torch 2.8, transformers 5.20). Training runs in `runs/`, exports in `exports/` (gitignored, only on this box). Final export: `exports/final_e17_s3`. |
| RTX 3060 (Arhan) | `ssh sra@10.1.205.86` (password: ask Arhan, never commit) | `~/Arhan/decide`, `conda activate decide` |
| Mac (Sahil) | local | `/Users/sahilapage/cognext`, the git checkout. A demo server may still be running on 127.0.0.1:8000. |

**Gotchas:**
* Never use `pkill -f <pattern>` over SSH (it kills your own session). Use `pgrep -f "[t]rain_mt.py"`.
* Start background jobs as `setsid nohup … < /dev/null & disown`.
* Find servers by port (`ss -ltnp | grep :8000`).
* The 4090 is shared with other users. Don't kill their processes.

## 4. Research survey status

Brief: survey the frontier work behind each part of Decide. Rank ideas by expected
gain per hour under the CPU <25 ms budget and the label-noise ceilings. Say what
survives a **different judge dataset**, propose a paper-level contribution, and
list what's hype.

| Track | Notes file | Status |
|---|---|---|
| Jev / Laya / industry ticket routing / LLM routers | `system_one_models_and_routing.md` | done |
| Small encoders, CPU inference, int8 fix, multi-task | `encoders_cpu_multitask.md` | done |
| Robustness to a different dataset | `robustness_distribution_shift.md` | done |
| Grounding / hallucination detectors (AlignScore, MiniCheck, HHEM, LettuceDetect…) | `grounding_detection.md` | was running at handover; **re-run if the file is missing** |
| Calibration under label noise and shift, conformal, noisy-label learning | `calibration_noisy_labels.md` | was running; **re-run if missing** |
| Selective prediction / abstention / cost-sensitive decisions (incl. the kNN-confidence idea) | `abstention_cost_decisions.md` | was running; **re-run if missing** |

After all six exist, write the combined report to
`reports/Decide frontier research survey.md`. Each idea should get its expected gain, cost
in hours, and how it holds up on a new dataset.

### Key findings so far (from the 3 finished tracks)

* **Jev and Laya are not competitors at our latency.**
  * Jev is hosted and takes 236–405 ms per call.
  * Laya is 421M parameters and takes seconds on a 4-core CPU. Its own card shows it ships over-confident (ECE 0.47 before a temperature refit), and its abstain head is broken (AUROC 0.30).
  * Pitch: 25× smaller, more than 10× faster on CPU, calibrated, with working abstention.
* **Size isn't the limit.** Flat quality from 17M to 150M means the ceiling is label noise.
  * The best use of the latency headroom is a 3-seed fp32 ensemble, averaged on logits before calibration, or Ettin-32M/68M.
  * Fancy multi-task optimisers don't beat a tuned weighted loss.
* **int8 broke grounded because of outlier activations.** If int8 is ever needed: static QDQ, per-channel weights, Percentile calibration, and keep the embeddings, FFN output, residual and LayerNorm layers in fp32.
* **Ettin cross-encoder rerankers (17M, May 2026)** are a candidate initialisation for the grounded decision.
* **For robustness to a new dataset:**
  * WiSE-FT / LP-FT style weight averaging toward the lightly-tuned model.
  * Learnable per-tier amount rules: a monotone model over log(amount) × tier.
  * Abstain when amount parsing fails or the amount is near a cap.
  * Self-test on held-out templates, LLM-paraphrased dev, CheckList invariance tests, and Bitext/BANKING77 mapped to the 5 queues.
  * Tune calibration and abstention on those shifted sets.
  * Skip test-time adaptation (TENT): it's unreliable on text and makes wrong answers over-confident. A label-prior EM re-estimate is the only cheap, safe adaptation.

## 5. Robustness session (2026-10-24, after the research) — what changed

* **Read the brief again, word for word.** Three things changed the plan:
  * The hidden test is the held-back ~20% of the **same** dataset.
  * "Data generation or collection" is out of scope, so no public datasets or LLM-written tickets in training.
  * "Pretraining a language model from scratch" is out of scope.
* **Research report:** `reports/Decide frontier research survey.md`, with the notes in `research_notes/`.
* **Seed ensemble:** no gain. The 4-seed average scores 84.0, against 84.8 for the old shipped seed (see `context.md` §7).
* **Stress test** (`tools/stress_test.py`): the same dev tickets, rewritten.
  * Route and approve are robust.
  * Grounded was brittle: typos 0.81, KB reorder 0.83, extra KB sentence 0.74.
  * Full write-up: `docs/robustness.md`.
* **Fixes:**
  * The number check now tolerates typos and ignores numbers copied from the ticket. Its features are identical on clean data.
  * Augmentation, grounded batches only (`--augment 2 --augment_tasks grounded`).
* **Shipped now:** `v6c2` seed 0.
  * Fresh-clone selfcheck **84.66**, all gates PASS, p95 6.3 ms.
  * Cross-fitted 83.8, the same as the old recipe's average seed.
  * Stress mean 83.0 vs 78.3.
  * The old model is tagged `v5-e17-s3`.
* **Ceiling analysis** (`tools/ceilings.py`, `docs/ceilings.json`):
  * Route error is flat at 5.5% across confidence.
  * Route abstention is 0.484 against a ceiling of 0.494.
* **Hand-written judge-style cases** (`tests/judge_cases.json`, `tools/judge_cases.py`):
  * The shipped model gets 21/26.
  * Grounded rejects correct hand-typed drafts on **new topics** with p ≈ 0.
  * Training articles always have 3–4 sentences; a one-sentence article is unfamiliar to the model.
* **In flight at handover**, both on the 4090, runs `runs/v6t2*` and `runs/v6L2*`:
  * `v6t2` adds the `kb_trim` augmentation family. The `kb_single` stress variant is the shipped model's weak spot (70.6).
  * `v6L2` is the same recipe, but initialised from `KRLabsOrg/tinylettuce-ettin-17m-en`: MIT, Aug 2025, the same Ettin-17M already fine-tuned for RAG hallucination detection. It should generalise better to new KB topics.
  * Evaluate both with `tools/eval_candidates.sh runs/<x>_e17 runs/<x>h_e17_s0 runs/<x>h_e17_s3`, which finalizes, stress-tests and runs the judge cases on the GPU box.
  * Ship only after a fresh-clone `selfcheck.py` over HTTP. Use a candidate branch, then fast-forward `main`.

## 6. Suggested next steps (in priority order)

1. **Pick between v6c2, v6t2 and v6L2** on three things:
   * cross-fitted clean score, which matters most because the hidden test is the same distribution
   * stress mean
   * judge cases
2. **Rehearse the 3-minute demo:**
   * one case that works
   * a correct abstention, e.g. "Do you have an office in Berlin?", which abstains on route
   * one trade-off, e.g. robustness augmentation for about 0 cost on the cross-fitted clean score
3. **Before judging:**
   * Ask the mentors whether the judge box has internet for `pip install`.
   * Give the judges the host:port.
   * Consider removing the internal IPs from `context.md`.

Rule kept all along: quote the cross-fitted number next to the in-sample one,
and never ship anything that hasn't gone through a full-dev `selfcheck.py`
over HTTP from a fresh clone.
