# context.md — everything done so far on "Decide"

> New session? Start with `HANDOVER.md` (current state, machines, next steps).

A running log of the hackathon work: what we picked, what we built, what we
tried, what worked, what failed, and where everything lives. Written so a
teammate (or a fresh Claude session) can pick up with no other context.

Repo: https://github.com/sahilapage/decide (branch `main`)

---

## 1. The hackathon and the choice

* 12-hour, 2-person hackathon with three problem statements:
  * **PS1** Smart Knowledge Base (RAG)
  * **PS2 "Decide"**: a CPU decision engine (this is the one we chose)
  * **PS3** CodeGuard
* **Why PS2:** it's scored by a fixed, automatic script (`selfcheck.py`), so
  every improvement can be measured objectively and we know the scoring. PS1 is
  judged on open-ended answer quality, retrieval and LLM behaviour, which is
  much harder to pin down in 12 hours.
* Compute:
  * **RTX 4090** over SSH (`sra@10.27.100.101`), working directory `~/sahil/decide`
  * also available: a 3060 and an H100 (not needed)
  * the Mac at `/Users/sahilapage/cognext` is the local copy and the git repo

## 2. The task (PS2 "Decide")

* **API:** one endpoint, `POST /v1/decide`. Each request asks for one of three
  decisions about a customer-support ticket:

  | Decision | Question | Output |
  |---|---|---|
  | **route** | which queue? | billing / technical / account / shipping / general |
  | **approve** | auto-approve the refund/credit, or send to a human? | yes/no; wrongly approving costs 3×, wrongly holding costs 1× |
  | **grounded** | is the drafted reply supported by the attached KB article? | yes/no |

* **Rules:**
  * CPU only, no network at inference
  * **one shared model** for all three decisions
  * p95 latency < 100 ms is a hard gate; full latency marks need ≤ 25 ms
  * artifacts under 500 MB
* **Score** (`selfcheck.py`):

  | Part | Weight | How it's measured |
  |---|---|---|
  | Quality | 40% | route macro-F1, approve cost vs always-hold, grounded macro-F1; scaled by coverage / 0.8 |
  | Calibration | 25% | ECE plus Brier skill |
  | Latency | 20% | p95 |
  | Abstention | 15% | `1 − risk@80% / full error`, ranked by confidence `max(p, 1−p)` |

  Approve cost score: `0.25 + 0.75·(hold − cost)/hold`.
* **Data:** 6,500 labelled train tickets (all three labels each) and 1,500 dev tickets.

## 3. What we learned from the data (this drove every design choice)

* **Route:**
  * About 5–6% of labels are random flips, e.g. "Package not delivered" labelled billing.
  * Vague tickets ("can someone help with my account") are labelled almost at random.
  * So no model can get much above **~90% route accuracy**.
* **Approve:**
  * Each tier has a soft amount cap: free ≈ $20–30, pro ≈ $100–120, enterprise ≈ $375–400.
  * Four disqualifying phrases.
  * About 20% of requests under the cap are refused for account-history reasons the ticket never mentions.
  * That puts the ceiling at **~91%**.
* **Grounded:**
  * Wrong drafts change exactly one fact (a "slot"): a number ("15" vs "30 minutes"), an audience ("admins" vs "all users") or a scope ("monthly" vs "annual", "all plans" vs "all paid plans").
  * Key discovery: after normalising numbers and units, **no correct draft ever states a number + unit (or range) that its KB article doesn't contain**.

## 4. The solution (what ships)

```
ticket + typed question
   → normalise numbers/durations ("two weeks" → "14 days", "a month" → "30 days")
   → short typed prompt ("route: …", "approve: …", "grounded: <draft> [SEP] <kb>")
   → ONE shared encoder: Ettin-17M (ModernBERT recipe, MIT), fine-tuned multi-task
   → mean-pool
   → three tiny heads: route Linear→5 | approve MLP([h, aux]) | grounded MLP([h, aux]) + learned flag weights
   → noise-aware calibration → decision rule → abstain policy
```

* **One model:** one ONNX graph with three outputs, served by ONNX Runtime fp32
  with 4 threads. More than 99.9% of the parameters are shared.
* **Grounded is a cross-encoder:** the draft and the KB article go through the
  encoder together as a pair, never embedded separately.
* **aux vector (71 dims):**
  * tier one-hot
  * tier × amount bucket
  * log amount
  * grounded features: "draft has a number fact not in the KB", how many, share supported, and "scope broadened" ("all N" vs "all <qualifier> N")
* **Learned flag weights:** `grounded_flag_w` and `grounded_scope_w` start at −8
  and are added straight to the grounded logit. The flag never fires on a
  correct draft, so the head doesn't have to rediscover the rule.
* **Normaliser** (`decide/normalize.py`):
  * number words, including hundreds and thousands
  * durations to base units: month = 30 days, year = 365 days, week = 7 days, hour = 60 minutes, quarter hour = 15 minutes
  * business days are kept as they are
* **Calibration** (noise-aware):
  * route: `p = (1−eps)·softmax(z/T + bias) + eps/5`. The label noise means even
    clear tickets are right only ~95% of the time, and temperature scaling alone
    can't express that ceiling.
  * binaries: `lo + (hi−lo)·sigmoid(a·z + b)`, bounded (A_MAX = 1.5, B_MAX = 5) so
    the almost-perfect grounded head can't produce exact 0/1 probabilities.
  * The noise-structure parameters are fit on 6,500 out-of-fold predictions;
    only the scales (T, a, b) are refit on the shipped model's dev logits.
* **Decision rules:**
  * approve = yes only if **p > 0.75**, where `3·(1−p) < p` under the 3:1 cost
  * grounded = yes if p > 0.5
  * route = the argmax queue
* **Abstention:**
  * Tuned per decision against an in-process copy of the judges' scorer.
  * Keeps the smallest abstain share within 0.002 of the best quality, with coverage at least 0.85.
  * For approve, the winning rule is the marginal cost effect `min(3(1−p), p) − λ·p`, which peaks near p = 0.75.

## 5. Final results (shipped model: Ettin-17M, seed 3, fp32, 68 MB)

* **Score:** dev selfcheck against the live service = **85.22 / 100**.

  | Part | Score |
  |---|---|
  | Quality | 0.852 |
  | Calibration | 0.846 |
  | Latency | 1.00 |
  | Abstention | 0.667 |

* **Honest estimate:** **84.7** when cross-fitted (policy fit on one half of
  dev, scored on the other half, 10 halves; range 82.6–86.2).
  * The in-sample number is slightly optimistic because the seed was picked on dev.
* **Per decision on dev:**

  | Decision | Accuracy | Answered-set quality | Coverage |
  |---|---|---|---|
  | Route | 88.3% | macro-F1 0.943 | 0.89 |
  | Approve | — | cost 0.191 (always-hold 0.374) | 0.85 |
  | Grounded | 100% | macro-F1 1.000 | 1.00 |

* **Out-of-fold on train (5-fold):** grounded 99.98%, route 89.2%, approve 89.1%.
  These are at the noise ceilings from section 3.
* **Latency (p95):**
  * 2.5 ms over HTTP on the Mac (Apple M5)
  * 5.4 ms in-process on 2 physical cores (i9-13900K, `taskset -c 0-3`)
  * 6.3 ms over HTTP in the clean-clone Linux test
* **Clean-clone test** (exactly what the judges will do):
  * fresh `git clone` on Linux, Python 3.10, CPU only
  * `./start.sh`, then `selfcheck.py`
  * result: **85.22**, 4,500 of 4,500 responses valid, all gates PASS

## 6. Score progression (what moved the needle)

| Step | Dev score |
|---|---|
| Starter-style sparse n-gram baseline (gate 1) | 77.0 |
| Ettin-32M multi-task transformer | 81.95 → 84.6 |
| Normaliser + number/scope flags | grounded 99.1% → 99.98% OOF |
| Noise-aware calibration | +0.65 (calibration part 0.832 → 0.858) |
| Short typed prompts, per-decision abstain tuning | small gains |
| Switch 32M → 17M | same quality, p95 13.4 → 5.4 ms on 2 cores |
| **Final** | **85.22** (cross-fitted 84.7) |

## 7. Things we tried that did NOT help (all measured)

* **int8 quantisation:**
  * ~2× faster, but −3 points overall: grounded accuracy fell from 99.1% to 92.0%.
  * Four int8 variants tested (`tools/quant_eval.py`): cross-fitted 80.0–80.4 vs 83.4 for fp32.
  * Comparing exact numbers needs precise activations. So we ship fp32 and use a smaller model for speed.
* **Distillation** from big teachers (Ettin 150M/400M, DeBERTa-v3-large, a Laya-initialised encoder): no gain.
* **Bigger models:** 150M ≈ 32M ≈ 17M in quality. The ceiling is label noise, not model size.
* **Model soups** (averaging weights of several seeds):
  * Broke badly when the head initialisation differed between seeds.
  * With a fixed head init (`--init_seed 1234`) they still hurt grounded (raw accuracy 96.2%).
  * So we ship a single seed.
* **Plain Newton Platt fit:** diverged on grounded. Replaced with bounded L-BFGS.
* **Stitched cross-fit** (two calibrations combined): too noisy. Each half is now scored on its own.

## 8. Infrastructure notes and gotchas

* **Remote env:** pip install of torch timed out, so the remote venv is built
  with `--system-site-packages` on top of `~/miniconda3/envs/trackeverything`
  (torch 2.8 cu129, transformers 5.20).
* **Weights:** Ettin and DeBERTa ship only `pytorch_model.bin`; fetched with `hf_hub_download`.
* **Killing processes:**
  * `pkill -f <pattern>` killed our own SSH session.
  * Use the bracket trick (`pgrep -f "[t]rain_mt.py"`) instead.
* **Background jobs over SSH:** run them as
  `setsid nohup … < /dev/null & disown`, or SSH hangs.
* **Shared GPU:** another user's job (PID 541579, a robotics training run) was on
  the 4090. Claude was not allowed to kill another user's process. The job later
  finished on its own.
* **GitHub file size:** GitHub's per-file limit is 100 MB.
  * The 128 MB 32M ONNX was split into 90 MB `model.onnx.part*` chunks, rejoined in memory at startup.
  * The 17M model fits in one 68 MB chunk.
* **Laptop latency** is noisy (p95 varies 3–8× between runs), so use a quiet
  desktop CPU (`taskset`) for final numbers.

## 9. Teammate's work (branch `perf/cpu-latency`) — merged

* Merged into `main` at commit `6e61016` with no conflicts. Re-verified afterwards: demo page HTTP 200, selfcheck 85.22.
* What it added:
  * CPU/GPU device switch in training (`--device auto`, `amp()`)
  * `DECIDE_SPIN` environment variable (thread spinning on/off; spin off removes rare p95 outliers)
  * `tools/latency_sweep.py`
  * `docs/benchmarks.md`
  * the offline demo page (`demo_ui.py` + `demo/index.html`)
* Their doc recommended int8. Our full-dev accuracy check showed it costs ~3
  points, so **we ship fp32**. The result is written up in `docs/benchmarks.md`.

## 10. Git history (main)

| Commit | What |
|---|---|
| `3e3ddc7` | Gate 1 baseline: sparse n-gram backbone, three heads, calibrated policy |
| `dab0cf1` | Gate 2: multi-task Ettin-32M student via ONNX Runtime on CPU |
| `248a2bd` | Short typed prompts, per-decision abstain tuning, Laya extractor, demo, docs drafts |
| `116f570` | (teammate) CPU latency work: device switch, spinning switch, sweep |
| `3e8c626` | (teammate) Live demo page |
| `f39ec0b` | Final model: Ettin-17M, dev 85.2 (cross-fitted 84.7), p95 2.5 ms |
| `6e61016` | Merge `perf/cpu-latency` |
| `717faed` | Benchmarks: int8 accuracy check + 17M latency |

## 11. File map

| Path | What |
|---|---|
| `app.py`, `selfcheck.py` | From the starter, unchanged |
| `model.py` | `DecisionModel`: loads `artifacts/student` + `policy.json`, falls back to `baseline.pkl` (`DECIDE_BACKEND=baseline`) |
| `decide/normalize.py` | Number/duration normaliser |
| `decide/textfmt.py` | Typed prompts (long/short), aux features, `number_facts()`, `scope_broadened()` |
| `decide/net.py` | `DecideNet`: shared encoder + three heads + flag weights |
| `decide/policy.py` | Calibration fits, `Policy`, decision rules, abstain tuning |
| `decide/scoring.py` | Offline copy of the selfcheck maths |
| `decide/student_runtime.py` | Rust tokenizer + ORT session; env vars `DECIDE_ONNX`, `DECIDE_THREADS`, `DECIDE_SPIN` |
| `decide/baseline.py`, `train_baseline.py` | Sparse baseline (dev ~81.7 with the new normaliser) |
| `train_mt.py` | Multi-task trainer (folds or `--full`, `--prompt`, `--soft`, `--ema`, `--seed`, `--init_seed`, `--device`) |
| `eval_runs.py` | OOF evaluation + structural policy fit |
| `fit_policy_dev.py` | Dev scale refit + abstain thresholds, cross-fitted estimate |
| `export_onnx.py` | ONNX export (fp32/int8), writes `meta.json` |
| `pack_student.py` | Chunks the model into `artifacts/student` |
| `finalize.sh` | Policy → dev policy → export → benchmark |
| `soup.py`, `bench_cpu.py` | Weight averaging, CPU latency benchmark |
| `tools/` | `crossfit_oof.py`, `quant_eval.py`, `extract_laya_encoder.py`, `latency_sweep.py` |
| `start.sh` | One-command serve: venv, pip install, uvicorn (uvloop/httptools) on 0.0.0.0:8000 |
| `demo.py`, `demo_ui.py`, `demo/index.html` | Terminal demo and web demo |
| `README.md`, `model_card.md`, `results.json`, `docs/benchmarks.md` | Submission docs |
| `artifacts/student/` | Shipped model: `model.onnx.part00` (68 MB), `meta.json`, `policy.json`, `tokenizer.json` |

## 12. How to run

```bash
./start.sh                                               # serve on :8000
python selfcheck.py --url http://localhost:8000/v1/decide
python demo.py                                           # terminal demo
.venv/bin/uvicorn demo_ui:app --port 8000                # web demo at http://localhost:8000/
```

Reproduce training (GPU box):

```bash
M=jhu-clsp/ettin-encoder-17m
python train_mt.py --model $M --out runs/v5_e17 --epochs 6 --lr 1e-4 --prompt short
python train_mt.py --model $M --out runs/v5h_e17_s3 --full --epochs 6 --lr 1e-4 --prompt short --seed 3
./finalize.sh runs/v5_e17 runs/v5h_e17_s3 exports/final_e17_s3
python pack_student.py exports/final_e17_s3 exports/final_e17_s3/policy.json
```

## 13. Likely judge questions (short answers)

* **"Is it really one model?"**
  * Yes: one encoder and one ONNX graph with three outputs.
  * The prompt names the question.
  * A new decision is one prompt string plus one small head.
* **"Why not 99% accuracy?"**
  * Route and approve labels carry built-in noise (random flips, hidden account history).
  * The ceilings are ~90% and ~91%, and we're at 88–89%.
  * Grounded, which has no such noise, is at 100% on dev.
* **"Why is it so fast?"**
  * A 17M encoder, fp32 ONNX Runtime, a Rust tokenizer, and short prompts.
  * p95 is 2.5–6.3 ms against a 25 ms target.
* **"Why not int8?"** Measured: it cuts grounded accuracy from 99% to 92%.
* **"How do you know it's not overfit to dev?"**
  * Network weights never see dev.
  * We report a cross-fitted 84.7 alongside the in-sample 85.2.
  * Out-of-fold numbers on train match.
* **"When does it abstain?"**
  * Route: vague tickets.
  * Approve: amounts near a tier's cap, where p is close to 0.75.
  * Grounded: essentially never.

## 14. Still to do

* Rehearse the 3-minute demo with `python demo.py` or the web page. It shows:
  * a confident route
  * a correct refusal
  * grounded catching one changed fact
  * approve at three amounts
* Give the judges the host:port where the service is running.
* Optional: benchmark `DECIDE_SPIN=0` vs `1` for the 17M model on 2 physical
  cores before changing the default (the current default is unchanged).

## 15. Arhan's side (optimisation, benchmarking, verification)

Added by Arhan's Claude Code session. Sections 1–14 above are Sahil's log and
stay the source of truth; this covers only what isn't there.

* **Independent verification of the shipped 17M (main @ 717faed)** on a
  second machine: Arhan's laptop (AMD Ryzen 7 7735HS, 8 cores / 16 threads, no
  GPU, Pop!_OS, Python 3.12).

  | Check | Result |
  |---|---|
  | Full dev `selfcheck.py` over HTTP | **85.22**, 4,500/4,500 valid, all gates PASS |
  | p95, 4 physical cores (`taskset -c 0,2,4,6`) | **4.3 ms** |
  | p95, 2 physical cores / 4 vCPU (`taskset -c 0-3`) | **6.3 ms** |

  The p95/p50 ratio is about 1.3, so the 17M is steady even on the noisy
  laptop, unlike the 32M.
* **Second GPU box (Arhan's):** `sra@10.1.205.86`, an RTX 3060 12 GB shared lab
  machine (password: ask Arhan; never commit it).
  * Clone: `~/Arhan/decide`, data copied into `data/`.
  * Env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate decide`
    (Python 3.11, torch 2.14.1+cu130, transformers 5.20.2, onnxruntime 1.31).
    Torch had to come from PyPI because `download.pytorch.org` reset the
    connection.
  * Ettin-17M trains in about 2 min per run (about 12 s/epoch).
* **Laptop training env:** `decide/.venv` also has CPU torch 2.14.1 and
  transformers 5.20.2, so `train_mt.py --device cpu` works there (slow, but no
  GPU needed).
* **Arhan's own 17M run** (long prompts, older policy code, offline dev 83.83)
  was only a size comparison and is superseded by Sahil's v5 17M. It lives in
  `runs/e17*` and `exports/e17` on both machines (gitignored, not used).
* **Deep research report** (outside the repo):
  `~/Final_Hackathon_PS_Decide/reports/Decide hackathon solution research.md`,
  with source notes in `research_notes/`. Verdicts:
  * Test-time training / adaptation: skip. There is no distribution shift, it
    costs at least one backward pass per request, and entropy-minimising TTA
    hurts calibration.
  * RL: skip. With full labels a bandit reduces to cost-sensitive
    classification, and the 3:1 cost has the closed-form threshold 0.75.
  * The only cheap "test-time compute" idea worth trying is kNN agreement over
    training embeddings as an extra confidence signal (not built).
* **Demo notes:**
  * A demo server started before `git pull` keeps serving the old model, so
    restart it after pulling.
  * Quote **84.7** (cross-fitted) as the expected score and 85.2 as in-sample.
* **Extra gotchas:**
  * `setsid cmd & echo $!` records setsid's PID, which exits immediately. Find a
    server by its port instead: `ss -ltnp | grep :8000`.
  * `/tmp` scratch files vanish on reboot, so keep anything worth having in the
    repo.
  * On SMT machines, check `lscpu -e=CPU,CORE` before trusting a "4-core"
    `taskset`.
