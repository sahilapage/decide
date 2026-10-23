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

* **Shipped model:** `artifacts/student/`, an Ettin-17M multi-task model, fp32 ONNX, 68 MB, seed 3.
* **Score:**
  * dev `selfcheck.py` = **85.22 / 100**
  * cross-fitted (honest) estimate = **84.7**
  * route acc 88.3%, approve cost 0.191 (always-hold 0.374), grounded 100%
* **Latency (p95):** 2.5 ms over HTTP on a Mac (M5), 4.3 ms on 4 physical cores, 6.3 ms on 2 physical cores.
* **Reproduced independently on two machines:**
  * a fresh Linux clone via `./start.sh`
  * Arhan's laptop
  * both gave 85.22, all gates PASS
* **Teammate branch:** `perf/cpu-latency` is merged. Nothing is pending on any branch.

Quick check from a fresh clone:

```bash
./start.sh &                         # serves :8000
python selfcheck.py --url http://localhost:8000/v1/decide   # expect ~85.2
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

## 5. Suggested next steps (in priority order)

1. **Finish the research:** re-run any missing track, then write the combined report.
2. **Build a "shifted dev" stress set:**
   * paraphrased tickets
   * held-out templates
   * amount-format variants
   * scaled caps
   * Bitext mapped to the 5 queues

   Measure how far the current model drops. This is the most important unknown.
3. ~~Seed ensemble~~: **done, no gain** (see `context.md` §7). Averaging the four 17M seeds scores 84.0, against 84.8 for the shipped seed. Don't spend more time here.
4. **Test kNN-agreement confidence for abstention** (teammate's idea). Abstention (0.667) is our weakest scoring part. Expect less than 1 point, because errors are mostly label noise.
5. **Robustness changes once the shifted set exists:** learnable amount rules, WiSE-FT averaging, and the Ettin reranker init for grounded.
6. **Before judging:**
   * Ask mentors whether the judge box has internet for `pip install`. If not, bundle wheels.
   * Rehearse `python demo.py` / `demo_ui.py`.
   * Give the judges the host:port.
   * Consider removing the internal IPs from `context.md`, since the repo is public.

Rule kept all along: never quote the in-sample 85.2 without the cross-fitted
84.7, and never ship anything that hasn't gone through a full-dev `selfcheck.py`
over HTTP.
