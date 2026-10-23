# Decide: one shared decision model, three typed answers, ~6 ms on CPU

`POST /v1/decide` answers three typed questions about a customer-support ticket:

| decision | question | answer |
|---|---|---|
| **route** | which queue should handle it? | one of billing / technical / account / shipping / general, with probabilities |
| **approve** | auto-approve the request? (a wrong approve costs 3× a wrong hold) | yes / no + `p_true` |
| **grounded** | is the draft reply supported by the KB article? | yes / no + `p_true` |

One fine-tuned 17M-parameter encoder answers all three in a single ONNX graph.
Probabilities are calibrated, every decision has a deliberate abstain
("hand to a human") policy, and everything runs on CPU only, with no network
calls and no API keys.

<!-- RESULTS:START -->
| dev split, via `selfcheck.py` against the live service | |
|---|---|
| route | acc 0.891, macro-F1 (answered) 0.946, coverage 0.85 |
| approve | acc 0.899, cost 0.201 vs always-hold 0.405 (3:1), coverage 0.90 |
| grounded | acc 1.000, macro-F1 1.000, coverage 1.00 |
| mean ECE / Brier skill | 0.014 / 0.821 |
| p95 latency (mean of 3 decisions) | 6.3 ms over HTTP on 4 logical CPUs (2 physical cores, i9-13900K); ~2.5 ms on an Apple M5 |
| **selfcheck total** | **85.9 / 100** (cross-fitted estimate 85.1, see model card) |
| shifted-dev stress test (typos, reordered/extra KB sentences, ticket refs, one-sentence KBs…) | held-out-shift mean 83.8 vs 82.5 for the un-augmented model; worst case 78 vs 65, see `docs/robustness.md` |
| hand-written tests | `tests/complex_cases.json` 29 / 31 (multi-issue tickets, quoted threads, mixed language, multi-claim drafts); `tests/judge_cases.json` 25 / 33 incl. 7 known-gap cases from an independent audit |
<!-- RESULTS:END -->

## Quick start

```bash
./start.sh                     # 1. serve the graded API on 0.0.0.0:8000 (creates .venv, installs, starts)
python test.py                 # 2. run the model over a whole dataset in-process: answers, timing, accuracy, score
python selfcheck.py --url http://localhost:8000/v1/decide   # 3. the judges' own check against the live service
```

### `test.py`: run any dataset, see every answer

```bash
python test.py                          # data/test.jsonl if it exists, else data/dev.jsonl
python test.py path/to/other.jsonl      # swap in any dataset in the same format
python test.py --limit 20               # first 20 records only
python test.py --quiet                  # summary only
```

No server and no port: it loads the shipped model directly and asks every
record all three decisions, in the same shuffled order the judges' harness
uses. It prints one line per ticket (each answer, its confidence, ABSTAIN when
it hands off, `ok`/`XX` against the label), then:
* **timing:** avg / p50 / p95 per decision
* **accuracy:** overall, on answered cases, and the abstain rate
* **total score:** the judges' formula, from the same metric code as `selfcheck.py`

`labels` are optional: without them you get answers and timing only.

Dataset format: one JSON object per line (a JSON list also works):

```json
{"request_id": "req-005178",
 "state": {"subject": "...", "body": "...", "customer_tier": "enterprise", "amount": 0.0,
           "draft_reply": "...", "kb_article": "..."},
 "labels": {"route": "billing", "approve": true, "grounded": true}}
```

### Serving details

Needs Python 3.10+ on a CPU-only Linux box; no GPU, and no network once it is
serving. The model weights are in the repo (`artifacts/student/`, 68 MB, stored
in <100 MB chunks that are joined in memory at startup).

* Serving installs only the five packages the request path imports (fastapi,
  uvicorn[standard], numpy, onnxruntime, tokenizers).
* If PyPI is unreachable, `start.sh` falls back to prebuilt Linux x86_64
  wheels for CPython 3.10–3.12 from this repo's `wheels-v1` release
  (`OFFLINE=1 ./start.sh` forces it).
* A restart with the packages already installed is up in about 2 s.
* Verified from fresh clones: online install (Python 3.10) and offline install
  (Python 3.11) both pass every gate; the current model scores 85.87.
* No `python3-venv` on a stock Debian/Ubuntu box? `start.sh` falls back to
  `virtualenv` (or run `apt install python3-venv`).

### Live demo page (optional, not part of the graded API)

```bash
python demo/demo_ui.py         # then open http://localhost:8000/
python demo/demo.py            # the same demo cases in the terminal
```

`demo/demo_ui.py` serves the same `/v1/decide` and `/health` from `app.py`
unchanged, and adds a single offline page (`demo/index.html`) with:
* preset demo cases and an editable ticket
* the three typed answers with abstain badges
* the gate-1 baseline's answer to the same ticket, for comparison
* an amount slider that re-asks approve live
* a live mini-evaluation: random dev tickets, clean and rewritten, through both
  models, scored with the judges' metric code
* a 200-call latency test and the raw JSON responses

## Repository layout

```
.
├── app.py              HTTP layer from the starter (fixed API contract)
├── model.py            DecisionModel: loads the shipped student + its policy
├── start.sh            one command to install and serve
├── test.py             run any dataset in-process: answers, timing, accuracy, score
├── selfcheck.py        the judges' check (schema, latency, score) against a live URL
├── decide/             the model code
│   ├── normalize.py        number / duration normaliser shared by all decisions
│   ├── textfmt.py          typed-question prompts and tabular features
│   ├── net.py              shared encoder + three heads
│   ├── policy.py           calibration, 0.75 approve rule, abstain tuning
│   ├── approve_stage.py    the approve head's tree layer
│   ├── student_runtime.py  tokenizer + ONNX Runtime session
│   ├── augment.py          label-preserving training augmentation
│   └── scoring.py          in-process copy of the selfcheck maths
├── artifacts/          the shipped model (student/: ONNX + tokenizer + policy) and the gate-1 baseline
├── training/           train_mt.py, finalize.sh, export_onnx.py, pack_student.py, eval_runs.py, ...
├── tools/              stress test, judge cases, latency benchmarks, ceiling and extensibility analysis
├── tests/              hand-written judge-style and complex-query cases
├── demo/               live demo page and terminal demo
└── docs/               robustness, ceiling, extensibility, benchmarks, demo script, team notes, research
```

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
        │        → 60 boosted trees           │
        │          (logit, amount÷cap, …)     │
   noise-aware    Platt w/ ceiling,       Platt, p > 0.5
   softmax        approve iff p > 0.75
        │             │                     │
   abstain if     abstain if p in         abstain only if
   low top-prob   (0.17, 0.83)            undecided
```

### Where the sharing happens

* `decide/textfmt.py`: every decision is a *(prompt, state)* pair fed to the
  same encoder. The prompt names the question, so one set of weights reads all
  three.
* `decide/net.py`: `DecideNet` has one `encoder` and three small heads (route
  Linear 1.3k parameters; approve and grounded MLPs ~84k each). About 99% of
  the parameters are shared.
* `artifacts/student/model.onnx`: **one graph, three outputs**. Every call runs
  the shared encoder once, and the API returns the head that was asked for.
  * The approve head's second layer, 60 boosted trees (~2.7k numbers), is
    compiled into the same graph as a standard `TreeEnsembleRegressor` node
    (`tools/embed_stage.py`).
  * Its inputs are the head's own logit plus amount ÷ tier cap, amount == 0
    and phrase flags.
  * It cannot run without the shared encoder's logit, and it leaves route and
    grounded untouched.
  * Without it, the same network scores 84.1 cross-fitted instead of 85.1.

**A fourth decision is cheap, and we measured it** ([docs/extensibility.md](docs/extensibility.md)).
* **Setup:** we trained on route + approve only, then added grounded afterwards
  as if it were new: one prompt string, one head, and **89 s** of training on
  top of the existing model, with a share of old-decision batches replayed.
* **Result:** **83.2** on dev against **83.6** for training all three together
  (cross-fit noise ±1.1). Grounded reached 99.9%, and route and approve kept
  their accuracy (within 1 point).
* **Head-only variant:** adding only the head on a frozen encoder keeps the
  other answers bit-identical, but reaches only 83% on grounded, because
  comparing two texts is a skill the encoder has to learn.

### Why each question is shaped differently

* **route** compares a ticket against five queues.
  * Multi-issue tickets mark the main issue with cue phrases ("the main thing
    I need resolved", "separately, less urgent"), and distractor sentences
    mention other queues' vocabulary, so a contextual encoder matters.
  * Labels are noisy by design: vague tickets ("can someone help with my
    account") get near-uniform probabilities and are abstained on.
* **grounded** compares *two texts*. Wrong drafts differ from the article in
  one slot: a number ("fifteen" vs "thirty minutes"), an audience
  ("administrators" vs "all users") or a scope ("monthly" vs "annual"). So the
  draft and the article go into the encoder **together** as a pair (a
  cross-encoder), never embedded separately.
  * Every duration is normalised to one unit using the data's own conventions
    ("a month" = 30 days, "six months" = 180 days, "a year" = 365 days,
    "a quarter of an hour" = 15 minutes, "one thousand" = 1000). Before this,
    most confident grounding errors were the encoder failing at unit
    arithmetic ("a month" vs "sixty days").
  * After normalisation, **no supported draft in train or dev states a
    number + unit (or a range like "3-5 business days") that its article does
    not**. Three number-consistency features (any unsupported number, how
    many, share supported) go into the grounded head next to the encoder
    output. The encoder handles the non-numeric slots.
* **approve** depends on more than words: per-tier amount caps and a few
  disqualifying phrases, plus account history the ticket never mentions.
  * The approve head sees the pooled text *and* a tabular vector (tier one-hot,
    tier × amount bucket, log amount).
  * Its second layer (60 boosted trees, `decide/approve_stage.py`, compiled
    into the ONNX graph) refines that logit with amount ÷ tier cap (caps
    estimated from train labels) and the disqualifying-phrase flags.
  * It is fitted on **out-of-fold** logits, so it corrects the network's real
    over-confidence (the network memorises the hidden-history refusals in
    train): +1.0 point cross-fitted.
  * Putting the same features into the network itself did not help (v7, `docs/report.md`).
  * With calibrated `p`, the 3:1 cost makes approving right only when
    `3(1−p) < p`, i.e. **p > 0.75**.

### Calibration and abstention

* **Calibration is label-noise aware.**
  * About 5–6% of routing labels are unrelated to the ticket text, so even the
    clearest ticket is right only ~95% of the time. Plain temperature scaling
    cannot express that plateau, so route uses
    `p = (1-eps)·softmax(z/T + bias) + eps/5`.
  * The binaries use Platt scaling with a learned floor and ceiling,
    `lo + (hi-lo)·sigmoid(a·z+b)`, because ~20% of approvable requests are
    refused for account-history reasons.
  * On 6,500 cross-fitted out-of-fold predictions this raises the calibration
    score from 0.832 to 0.858 (total +0.65).
* **What is fitted where.** The structural noise parameters (eps, bias, floor,
  ceiling) are fitted on the 6,500 out-of-fold predictions. Only the scales
  (T, a, b) are refitted on the shipped model's own dev predictions.
  * So the headline selfcheck number is in-sample for those few parameters.
  * Next to it we report the cross-fitted estimate (fit on one half of dev,
    score the other).
  * Dev also guided choices beyond the policy: the seed (best of two), the
    recipe among the candidates in `docs/report.md`, and feature checks. So
    treat ~84.5–85 as the expectation on unseen data.
* **Abstention is tuned per decision** against an in-process copy of the
  judges' scorer (`decide/scoring.py`, built from `selfcheck.py`'s own
  functions).
  * For each decision we search abstain shares up to 15% (coverage floor
    0.85, a margin above the 80% target) and keep the smallest share that
    reaches the best quality.
  * Three approve rules are searched: by cost, by confidence, and a cost-ratio
    rule. The shipped `policy.json` uses the confidence rule: abstain when p is
    between 0.17 and 0.83, i.e. the least certain ~10% of requests.
  * Route abstains on the ~15% lowest top-probability tickets; grounded
    essentially never.

## Reproduce

```bash
pip install -r requirements-train.txt                                # GPU box
python training/train_baseline.py                                    # gate-1 sparse baseline
M=jhu-clsp/ettin-encoder-17m
C="--model $M --epochs 6 --lr 1e-4 --prompt short --augment 2 --augment_tasks grounded"
python training/train_mt.py $C --out runs/v6t2_e17                       # 5-fold OOF
python training/train_mt.py $C --out runs/v6t2h_e17_s3 --full --seed 3  # shipped network
python tools/fit_approve_stage.py runs/v6t2_e17 runs/v6t2h_e17_s3        # approve stage on OOF logits
./training/finalize.sh runs/v6t2_e17_st runs/v6t2h_e17_s3_st exports/v6t2h_s3_st   # calibrate, export, benchmark
python tools/embed_stage.py exports/v6t2h_s3_st --check data/dev.jsonl   # trees into the ONNX graph
python training/pack_student.py exports/v6t2h_s3_st exports/v6t2h_s3_st/policy.json
python tools/stress_test.py                                              # shifted-dev robustness check
```

Run every command from the repository root. `training/train_baseline.py`
takes no arguments and overwrites `artifacts/baseline.pkl`.

## Pretrained components and licences

| Component | Used for | Licence |
|---|---|---|
| [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | **shipped backbone** | MIT |
| [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) | compared, same quality, 2.7x slower | MIT |
| [jhu-clsp/ettin-encoder-150m / -400m](https://huggingface.co/jhu-clsp) | teachers (training experiments only) | MIT |
| [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) | teacher (training only) | MIT |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) encoder | teacher init (training only) | Apache-2.0 (per its model card) |
| [KRLabsOrg/tinylettuce-ettin-17m-en](https://huggingface.co/KRLabsOrg/tinylettuce-ettin-17m-en) | initialisation experiment (not shipped) | MIT |

**Libraries:**
* ONNX Runtime (MIT), ONNX (Apache-2.0), Hugging Face tokenizers (Apache-2.0)
* FastAPI (MIT), uvicorn (BSD-3), NumPy (BSD-3)
* training only: transformers (Apache-2.0), PyTorch (BSD-3), scikit-learn (BSD-3), SciPy (BSD-3)

**Training data:** only the provided train split. `decide/augment.py` adds
label-preserving rewrites of provided train rows (typos, reordered or trimmed
KB sentences, polite opener/closer sentences) for grounded batches only. That
is augmentation, not data generation: no new labels, no external or LLM data,
and dev is never augmented.

Versions we ran: Python 3.11, onnxruntime 1.23/1.31, tokenizers 0.22,
fastapi 0.161, torch 2.8 (training), transformers 5.20 (training).

## Further reading

| Doc | What it covers |
|---|---|
| `model_card.md` | intended use, metrics, limits |
| `docs/robustness.md` | stress tests on shifted text, and the fixes |
| `docs/ceiling.md` | how high this data lets any model score (~87), and why |
| `docs/extensibility.md` | adding a fourth decision, measured |
| `docs/benchmarks.md` | CPU latency, and why int8 is not shipped |
| `docs/demo_script.md` | the three-minute demo and likely questions |
| `docs/report.md` | decision matrix, experiment tracker, error analysis |
