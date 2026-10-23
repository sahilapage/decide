# Decide — starter repo

A working, schema-valid endpoint that scores almost nothing. Your job is to make
it good.

Everything here exists so that you never lose time to HTTP plumbing, JSON shapes
or measurement. None of it does any of the thinking for you.

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide --limit 200
```

That should run clean and give you a very low score. You now have a valid
submission. Everything from here is improvement, not integration.

Put `train.jsonl` and `dev.jsonl` in `data/`.

## The only file you change

**`model.py`.** It holds one class with three methods:

```python
route(state)     -> (value, probs, confidence, abstain)
approve(state)   -> (value, p_true, abstain)
grounded(state)  -> (value, p_true, abstain)
```

Keep those signatures and return shapes and the rest of the repo keeps working.

`app.py` is the HTTP layer and implements the required contract exactly. You
should not need to touch it. If you find yourself changing the response shape,
stop — automated judging will reject it.

`selfcheck.py` is the same scoring code the judges run, pointed at your dev
split. There is no hidden extra criterion.

## What you have to decide

All of it. The model, the features, how the three decision types share a
backbone, how you train, how you turn scores into probabilities you can defend,
and when to abstain. That is the exercise, and the scoring rubric is the only
guidance you get.

Four things worth knowing before you pick an approach:

- **A simple baseline is harder to beat than you expect.** Build one first. It
  takes twenty minutes and tells you how hard the problem actually is.
- **The three questions are not all the same shape.** Think about what each one
  is comparing before you commit to an architecture.
- **Raw model confidence is usually overconfident.** Accuracy and calibration
  are separate problems with separate fixes. `selfcheck.py` reports ECE.
- **Look at the trivial-policy lines in the report.** For at least one decision,
  doing nothing at all is a respectable strategy, and your threshold probably
  should not be 0.5.

## Gates

| | By | What |
|---|---|---|
| Gate 1 | T+2:00 | A simple baseline submitted and scored |
| Gate 2 | T+5:30 | A live endpoint conforming to the schema, serving *something* |

Gate 2 is why the stub exists. Get a service running early even if the model
inside it is useless, then improve the model. Teams that stay in a notebook
until hour nine do not finish.

## Hard constraints

- **CPU only at inference.** Train on GPU if you like; serve on CPU.
- **No network calls while serving.** No LLM API, no hosted embeddings, nothing
  external in the request path.
- **p95 under 100ms per call**, measured by the judges on a 4-core machine.
  Full marks at 25ms.
- **One shared model across all three decision types**, not three separate
  models behind a router. Your README must say where the sharing happens.
- **Artifacts under 500MB total.**
- Pretrained open-weight models published before the event are fine. Name each
  one and its licence in your README.

## Reading your self-check

| Axis | Weight | What it is |
|---|---|---|
| quality | 40% | How often you are right, scored against the best trivial policy for each decision |
| calibration | 25% | Whether your probabilities mean what they say (ECE, Brier) |
| latency | 20% | p95 per call. Full marks at 25ms, zero at the 100ms gate |
| abstention | 15% | Whether your confidence ranks your own errors, at 80% coverage |

Your latency numbers depend on your machine. The judges re-measure on theirs, so
treat yours as a guide. Measure one request at a time — batching many requests
and dividing is not latency and will not be accepted.

## Submitting

- Public Git repo: code, `README.md`, `model_card.md`, `results.json`
- A **single command** in your README that starts the service on a clean 4-core
  Linux box. If the judges cannot start it, it cannot be scored.
- The host and port where your service is running at the deadline
- `results.json` — produced by `python selfcheck.py --url ...`
- `model_card.md` — one page: what you built, what you trained on, your measured
  accuracy and calibration, where it fails, and when it abstains. **Submissions
  without a model card are not ranked.**

If your weights are not in the repo, your README must say how to produce or
fetch them. A service the judges cannot start scores zero.

## Practical

- **Commit and push every thirty minutes.** Free Lightning Studios restart every
  four hours. That will happen to you twice. Anything unpushed is gone.
- **Pre-download any pretrained weights** before the clock starts.
- Mentors will help with environment, tooling and debugging. They will not tell
  you what architecture to use or why a decision is scoring badly.

## What is deliberately not in here

No model, no features, no training script, no calibration, no abstention policy,
no serving or threading configuration. Those are the parts being assessed. If
something feels missing, it is probably the thing you are supposed to build.
