# Decide — one shared decision model, three typed answers, ~3 ms on CPU

`POST /v1/decide` answers three typed questions about a support ticket —
**route** (5-way choice), **approve** (yes/no under a 3:1 cost) and
**grounded** (is the draft supported by the article) — with **one** fine-tuned
encoder, calibrated probabilities and a deliberate abstain policy, on CPU only,
with no network calls.

<!-- RESULTS:START -->
| dev split, via `selfcheck.py` against the live service | |
|---|---|
| route | acc 0.891, macro-F1 (answered) 0.946, coverage 0.85 |
| approve | acc 0.890, cost 0.204 vs always-hold 0.368 (3:1), coverage 0.85 |
| grounded | acc 1.000, macro-F1 1.000, coverage 1.00 |
| mean ECE / Brier skill | 0.012 / 0.811 |
| p95 latency (mean of 3 decisions) | 6.3 ms over HTTP on 4 logical CPUs (2 physical cores, i9-13900K); ~2.5 ms on an Apple M5 |
| **selfcheck total** | **85.0 / 100** (cross-fitted estimate 84.1, see model card) |
| shifted-dev stress test (typos, reordered/extra KB sentences, ticket refs, one-sentence KBs…) | held-out-shift mean 83.8 vs 82.5 for the un-augmented model; worst case 78 vs 65, see `docs/robustness.md` |
| hand-written judge-style cases (`tests/judge_cases.json`) | 25 / 26 as expected (the miss abstains) |
<!-- RESULTS:END -->

## Start it (one command)

```bash
./start.sh                  # creates .venv, installs requirements.txt, serves on 0.0.0.0:8000
```

Then `python selfcheck.py --url http://localhost:8000/v1/decide`. Needs Python
3.10+ on a CPU-only Linux box; no GPU, and no network once it is serving. The
model weights are in the repo (`artifacts/student/`, 68 MB, stored in <100 MB
chunks that are joined in memory at startup).

* Serving installs only the five packages the request path imports (fastapi,
  uvicorn[standard], numpy, onnxruntime, tokenizers).
* If PyPI is unreachable, `start.sh` falls back to prebuilt Linux x86_64
  wheels for CPython 3.10–3.12 from this repo's `wheels-v1` release
  (`OFFLINE=1 ./start.sh` forces it).
* A restart with the packages already installed is up in about 2 s.
* Verified from fresh clones: online install on Python 3.10 and offline install
  on Python 3.11 both score 84.98 with `selfcheck.py`.

### Live demo page (optional, not part of the graded API)

```bash
.venv/bin/uvicorn demo_ui:app --host 0.0.0.0 --port 8000   # then open http://localhost:8000/
```

`demo_ui.py` serves the same `/v1/decide` and `/health` from `app.py`
unchanged and adds a single offline page (`demo/index.html`): preset demo
cases, an editable ticket, the three typed answers with abstain badges, an
amount slider that re-asks approve live, a 200-call latency test and the raw
JSON responses. `python demo.py` runs the same cases in the terminal.

## How it works

```
             state + typed question
                      │
     normalise numbers/durations ("two weeks" → "14 days")
                      │
   "route: <subject> | <body>"            ┐
   "approve: <tier> $<amount> | ..."      ├─ one prompt per decision
   "grounded: <draft>" [SEP] "<kb>"       ┘
                      │
          ┌───────────▼────────────┐
          │  shared encoder         │   Ettin-17M (ModernBERT recipe), fine-tuned
          │  (one ONNX graph)       │   multi-task on all three decisions
          └───────────┬────────────┘
                 mean-pool h
        ┌─────────────┼─────────────────────┐
   route head     approve head            grounded head
   Linear→5     MLP([h, aux])          MLP([h, aux]) + w·flags
        │             │                     │
   temperature     Platt, approve iff     Platt, p > 0.5
   scaling         p > 0.75 (3:1 cost)
        │             │                     │
   abstain if     abstain where it        abstain only if
   low top-prob   most hurts 1-cost/hold  undecided
```

### Where the sharing happens

* `decide/textfmt.py` — every decision is a *(prompt, state)* pair fed to the
  same encoder. The prompt names the question, so one set of weights reads all
  three.
* `decide/net.py` — `DecideNet`: one `encoder`, three heads of a few thousand
  parameters each. More than 99.9% of the parameters are shared.
* `artifacts/student/model.onnx` — one graph, three outputs. Every call runs
  the shared encoder once; the API returns the head that was asked for.

A fourth decision is one more prompt string and one more linear head (~400
parameters for a binary question), trained on top of the same encoder.

### Why each question is shaped differently

* **route** compares a ticket against five queues. Multi-issue tickets mark the
  main issue with cue phrases ("the main thing I need resolved", "separately,
  less urgent"), and distractor sentences mention other queues' vocabulary, so
  a contextual encoder matters. Labels are noisy by design: vague tickets
  ("can someone help with my account") get near-uniform probabilities and
  are abstained on.
* **grounded** compares *two texts*. Wrong drafts differ from the article in
  one slot — a number ("fifteen" vs "thirty minutes"), an audience
  ("administrators" vs "all users") or a scope ("monthly" vs "annual"). So the
  draft and the article go into the encoder **together** as a pair (a
  cross-encoder), never embedded separately.
  * Every duration is normalised to one unit using the data's own conventions
    ("a month" = 30 days, "six months" = 180 days, "a year" = 365 days,
    "a quarter of an hour" = 15 minutes, "one thousand" = 1000). Before this,
    most confident grounding errors were the encoder failing at unit
    arithmetic ("a month" vs "sixty days").
  * After normalisation, **no supported draft in train or dev states a
    number + unit (or range like "3-5 business days") that its article does
    not**. Three number-consistency features (any unsupported number, how
    many, share supported) go into the grounded head next to the encoder
    output. The encoder handles the non-numeric slots.
* **approve** depends on more than words: per-tier amount caps and a few
  disqualifying phrases, plus account history the ticket never mentions. The
  approve head sees the pooled text *and* a tabular vector (tier one-hot,
  tier × amount bucket, log amount). With calibrated `p`, the 3:1 cost makes
  approving right only when `3(1−p) < p`, i.e. **p > 0.75**.

### Calibration and abstention

* Calibration is label-noise aware. About 5-6% of routing labels are
  unrelated to the ticket text, so even the clearest ticket is right only ~95%
  of the time; plain temperature scaling cannot express that plateau. Route
  uses `p = (1-eps)·softmax(z/T + bias) + eps/5`; the binaries use Platt
  scaling with a learned floor and ceiling (`lo + (hi-lo)·sigmoid(a·z+b)`),
  because ~20% of approvable requests are refused for account-history reasons.
  On 6,500 cross-fitted out-of-fold predictions this raises the calibration
  score from 0.832 to 0.858 (total +0.65).
* The structural noise parameters (eps, bias, floor, ceiling) are fitted on
  the 6,500 out-of-fold predictions; only the scales (T, a, b) are refitted on
  the shipped model's own dev predictions, and the dev numbers we report are
  cross-fitted (fit on one half of dev, score the other).
* Abstention is tuned per decision against an in-process copy of the judges'
  scorer (`decide/scoring.py`, built from `selfcheck.py`'s own functions):
  for each decision we search abstain shares up to 18.5% and keep the
  smallest share that reaches the best quality, so coverage stays above 80%.
  For approve, the rule that wins is the marginal effect of a case on
  `1 − cost/hold_cost`: `min(3(1−p), p) − 0.9·p`, which peaks near p = 0.75.

## Reproduce

```bash
pip install -r requirements-train.txt                       # GPU box
python train_baseline.py                                    # gate-1 sparse baseline
M=jhu-clsp/ettin-encoder-17m
C="--model $M --epochs 6 --lr 1e-4 --prompt short --augment 2 --augment_tasks grounded"
python train_mt.py $C --out runs/v6t2_e17                       # 5-fold OOF
python train_mt.py $C --out runs/v6t2h_e17_s3 --full --seed 3  # shipped network
./finalize.sh runs/v6t2_e17 runs/v6t2h_e17_s3 exports/v6t2h_s3   # calibrate, export, benchmark
python pack_student.py exports/v6t2h_s3 exports/v6t2h_s3/policy.json
python tools/stress_test.py                                     # shifted-dev robustness check
```

## Pretrained components and licences

| Component | Used for | Licence |
|---|---|---|
| [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | **shipped backbone** | MIT |
| [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | compared, same quality, 2.7x slower | MIT |
| [jhu-clsp/ettin-encoder-150m / -400m](https://huggingface.co/jhu-clsp) | teachers (training experiments only) | MIT |
| [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) | teacher (training only) | MIT |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) encoder | teacher init (training only) | Apache-2.0 |

Libraries: ONNX Runtime (MIT), Hugging Face tokenizers (Apache-2.0),
transformers (Apache-2.0), PyTorch (BSD-3), FastAPI (MIT), uvicorn (BSD-3),
scikit-learn (BSD-3), SciPy (BSD-3), NumPy (BSD-3).

Versions we ran: Python 3.11, onnxruntime 1.23/1.31, tokenizers 0.22,
fastapi 0.161, torch 2.8 (training), transformers 5.20 (training).

## Repo map

| Path | What |
|---|---|
| `app.py` | HTTP layer from the starter (unchanged contract) |
| `model.py` | `DecisionModel`: loads the student (or the baseline) + policy |
| `decide/normalize.py` | number / duration normaliser shared by all decisions |
| `decide/textfmt.py` | typed-question prompts and tabular features |
| `decide/net.py` | shared encoder + three heads |
| `decide/policy.py` | calibration, 0.75 approve rule, abstain tuning |
| `decide/scoring.py` | offline copy of the selfcheck maths |
| `decide/student_runtime.py` | tokenizer + ONNX Runtime session |
| `train_mt.py`, `eval_runs.py`, `export_onnx.py`, `pack_student.py` | training pipeline |
| `bench_cpu.py`, `demo.py` | latency benchmark, live demo |
| `model_card.md`, `results.json` | submission artefacts |
