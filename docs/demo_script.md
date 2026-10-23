# Three-minute demo: run of show

The brief asks for three things: show it working, show a case where it
correctly refuses to answer, and explain one trade-off.

## Before the judge arrives

* Start the service: `./start.sh`, or `.venv/bin/uvicorn demo_ui:app --port 8000` for the web page.
* Check it is up: `curl localhost:8000/health` should return `{"status":"ok","loaded":true}`.
* Run `python demo.py` once to warm it up.

## 0:00 – 0:30 · What it is

> "Decide is a System One decision model like Jev or Laya, but 25× smaller than Laya: 17M parameters on CPU, about 6 ms per call. One shared encoder answers three typed questions. Each answer comes with a calibrated probability and an abstain flag."

## 0:30 – 1:30 · Show it working (`python demo.py`, cases 1, 3, 3b, 5, 5b)

* **Case 1, one model:** the same ticket gets route, approve and grounded from one model, each in a few ms.
* **Cases 3 and 3b, grounding catches one changed fact:**
  * The article says *annual* subscribers and the draft says *monthly*, so p ≈ 0.
  * Fix the article and p ≈ 1.
* **Cases 5 and 5b, an off-template ticket typed by hand:**
  * A one-sentence article, a "thanks for getting in touch" opener and an order number in the draft: still grounded, p ≈ 1.
  * Change "5 days" to "10 days" and p ≈ 0.

> "Grounding is a cross-encoder: the draft and the article are read together. Numbers and durations are normalised first, so 'two weeks' equals '14 days'."

## 1:30 – 2:15 · A correct refusal (case 2, then case 4)

* **Case 2:** "Can someone help with my account please" gives a near-flat queue distribution (top probability about 0.34), and the model abstains.
* **Case 4:** the same refund on the Free tier at $8, $22 and $60:
  * $8 is approved (p 0.84).
  * $22 abstains: p 0.61 is below the 0.75 break-even of the 3:1 cost, and the model is unsure, so a human decides.
  * $60 is held (p ≈ 0).

> "The 3:1 cost means we approve only when p > 0.75, because 3·(1−p) < p. Near the cap we hand the case to a human."

## 2:15 – 3:00 · One trade-off, pick one

* **int8 vs fp32.**
  * int8 halves latency, but it dropped grounded accuracy from 99% to 92%. Exact number comparison needs precise activations.
  * We are at 6 ms against a 25 ms target for full marks, so speed would buy nothing.
  * We ship fp32.
* **Robustness vs dev score.**
  * We rewrote dev tickets with typos, reordered articles, one-sentence articles and order IDs. The template-trained model fell to 65 on the worst variant; now its worst is 78.
  * The cost is about 0 on the cross-fitted dev score.
  * We report the cross-fitted 84.1 next to the in-sample 85.0, because picking a seed on dev is optimistic.
* **Noise ceiling.**
  * Route errors are flat at 5.5% even on the model's most confident cases: those labels are random.
  * Route abstention sits 0.01 below its mathematical ceiling, so we stopped tuning it.

## Likely questions

| Question | Answer |
|---|---|
| "Is it really one model?" | Yes. One encoder and one ONNX graph with three outputs; the prompt names the question. A fourth decision is one prompt and a small head (`decide/net.py`). |
| "Why not 99% on route?" | About 5–6% of route labels are random (the brief says so). Our most confident 10% are still wrong 5.5% of the time, which is the floor. |
| "Why not RL or test-time training?" | Every example is labelled, so RL reduces to cost-sensitive classification, and the 0.75 threshold is the exact optimum. Test-time adaptation makes wrong answers more confident. |
| "Overfit to dev?" | Network weights never see dev. Dev only fits 3 calibration scales and the abstain thresholds. We report the cross-fitted number. |
| "What if the data changes?" | `docs/robustness.md` covers 13 shifted variants plus 26 hand-written tickets (25/26 correct; the miss abstains). |
