# How high can the score go on this data?

Shipped model: **85.87** on the judges' check (`selfcheck.py`, fresh clone),
85.1 cross-fitted. This page estimates the best score *any* model could reach on
this dataset, part by part, from measurements on our own predictions.

**Short answer: about 87. A score of 90 is not available.** It would need the
noise in the labels to be predictable, and we tested that directly: it isn't.

| part (weight) | ours | best possible | what caps it |
|---|---|---|---|
| latency (20%) | 1.000 → 20.0 | 20.0 | full marks at p95 ≤ 25 ms; we're at 6.3 ms |
| quality (40%) | 0.856 → 34.3 | ≈ 0.865 → 34.6 | route labels with random flips; approve refusals driven by hidden account history; grounded already 1.0 |
| calibration (25%) | 0.876 → 21.9 | ≈ 0.885 → 22.1 | ECE can't fall below a 1,500-row sample's own noise; Brier skill is capped by the same label noise |
| abstention (15%) | 0.647 → 9.7 | ≈ 0.67 → 10.0 | route is at its floor; approve has a little room; grounded is already 1.0 |
| **total** | **85.9** | **≈ 86.7–87** | |

## The evidence, part by part

### Route: about 5.5% of labels are random
* Out of fold (6,500 rows, `tools/ceilings.py`), the error rate among the model's most confident 10%, 20%, 40%, 60% and 80% of tickets is 5.5%, 5.5%, 5.5%, 5.6% and 5.7%. The curve is flat, so no confidence signal can rank those errors.
* Route risk@80% on dev is 0.0550, against a full error of 0.1093, giving an abstention part of **0.497**. The out-of-fold ceiling is 0.494, so route is at its floor.
* Answered-set macro-F1 is 0.946. With 5.5% of answered labels random, F1 cannot rise much further.
* Every recipe we tried has the same route accuracy (88.9–89.5%): encoders from 17M to 150M parameters, five training losses for noisy labels, and a pretrained hallucination detector as the starting point.

### Approve: about 15% hidden refusals
* Below the cap with no disqualifying phrase, 15% of requests are refused.
* Nothing in the text predicts which: an n-gram model scores AUC **0.46** and the network **0.52** (`tools/approve_stack.py`). That is the "account history the ticket does not mention" from the brief.
* The cost score has a closed-form limit. For the requests a perfect model would approve, the ratio cost ÷ hold-cost is 3h/(1−h), where h is the hidden refusal rate: 3·0.15/0.85 ≈ 0.53. Abstaining doesn't change that ratio, because removing cases scales both the cost and the hold-cost alike.
* Only the gradual fall-off near each tier's cap leaves room to do better than 0.53. We are at 0.201/0.405 = 0.50, so the cost score is 0.63.
* The approve second stage (`decide/approve_stage.py`) took accuracy from 88.5% to 90.3% out of fold, which is most of the remaining headroom.

### Calibration: at the sampling floor
We drew labels from our own probabilities 300 times, so the model is
calibrated by construction, and measured the ECE the judges' estimator gives on
1,500 rows:

| | ECE of a perfectly calibrated model (mean, 10–90%) | ours |
|---|---|---|
| route | 0.018 (0.013–0.024) | 0.024 |
| approve | 0.014 (0.009–0.020) | 0.017 |

At most about +0.2 points is left in calibration.

### Grounded: perfect
Accuracy 1.000, ECE 0.000 and Brier skill 1.000, with abstention 1.0.

## What this means for ranking

Every team works on the same labels, so every team hits the same ceiling. The
remaining differences between teams come from:
* how close each decision gets to its floor (ours: route at the floor; approve and calibration within a few tenths)
* latency (full marks for us)
* robustness if the hidden set is noisier or phrased differently (`docs/robustness.md`)
