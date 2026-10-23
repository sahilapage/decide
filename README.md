# Decide — one shared decision model, three typed answers, ~5 ms on CPU

`POST /v1/decide` answers three typed questions about a support ticket —
**route** (5-way choice), **approve** (yes/no under a 3:1 cost) and
**grounded** (is the draft supported by the article) — with **one** fine-tuned
encoder, calibrated probabilities and a deliberate abstain policy, on CPU only,
with no network calls.

<!-- RESULTS:START -->
| | dev (1,500 tickets, 4,500 calls) |
|---|---|
| route | acc 0.893, macro-F1 (answered) 0.944 |
| approve | cost 0.211 vs always-hold 0.382 (3:1) |
| grounded | acc 0.997, macro-F1 (answered) 1.000 |
| mean ECE | 0.023 |
| p95 latency | 5.5 ms (Apple M5, 4 threads) / 15.8 ms (i9-13900K pinned to 4 cores) |
| selfcheck total | 84.6 / 100 |
<!-- RESULTS:END -->

## Start it (one command)

```bash
./start.sh                  # creates .venv, installs requirements.txt, serves on 0.0.0.0:8000
```

Then `python selfcheck.py --url http://localhost:8000/v1/decide`. Needs Python
3.10+ on a CPU-only Linux box; no GPU, no network after `pip install`. The
model weights are in the repo (`artifacts/student/`, stored as two <100 MB
chunks that are joined in memory at startup).

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
          │  shared encoder         │   Ettin-32M (ModernBERT recipe), fine-tuned
          │  (one ONNX graph)       │   multi-task on all three decisions
          └───────────┬────────────┘
                 mean-pool h
        ┌─────────────┼─────────────────────┐
   route head     approve head            grounded head
   Linear→5     MLP([h, tier×amount])      Linear→1
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
  cross-encoder), never embedded separately, and numbers are normalised first.
* **approve** depends on more than words: per-tier amount caps and a few
  disqualifying phrases, plus account history the ticket never mentions. The
  approve head sees the pooled text *and* a tabular vector (tier one-hot,
  tier × amount bucket, log amount). With calibrated `p`, the 3:1 cost makes
  approving right only when `3(1−p) < p`, i.e. **p > 0.75**.

### Calibration and abstention

* Probabilities are calibrated per decision (temperature for route, Platt
  scaling for the binaries) on **out-of-fold** predictions from 5-fold
  training, never on dev.
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
python train_mt.py --model jhu-clsp/ettin-encoder-32m --out runs/e32s --prompt short        # 5-fold OOF
python train_mt.py --model jhu-clsp/ettin-encoder-32m --out runs/e32s_full --full --prompt short
python eval_runs.py runs/e32s --save_policy runs/e32s_policy.json
python export_onnx.py runs/e32s_full exports/e32s
python pack_student.py exports/e32s runs/e32s_policy.json
python bench_cpu.py exports/e32s                            # latency + ONNX/PyTorch parity
```

## Pretrained components and licences

| Component | Used for | Licence |
|---|---|---|
| [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | shipped student backbone | MIT |
| [jhu-clsp/ettin-encoder-150m / -400m](https://huggingface.co/jhu-clsp) | teachers (training only) | MIT |
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
