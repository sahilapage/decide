# Calibration under label noise & distribution shift, conformal prediction, and noisy-label learning — for "Decide" (Ettin-17M, route 5-way / approve / grounded)

*Verification key: **[V]** = content checked against the primary source (or its abstract page) during this session (2026-10-24). **[K]** = primary-source citation from the researcher's prior knowledge of the paper; the link is the paper, but the specific claim was not re-read this session. Treat [K] numbers with more caution. Lines marked "Inference" apply the literature to Decide; they are not published results.*

## 1. Post-hoc calibration methods: which is best for small calibration sets (1.5k–6.5k) and 5 classes?

### Takeaway
On small calibration sets, methods with few parameters that keep accuracy unchanged (temperature scaling and its extensions: TS plus bias, ensemble TS with a uniform component, Platt/Beta for binaries) are the most reliable. More flexible maps (matrix/Dirichlet, isotonic, histogram binning, splines) need more data, and their gains mostly appear with many classes or large calibration sets. Decide's current maps (TS + bias + eps-uniform floor for route; bounded Platt for the binaries) already match "ensemble temperature scaling" and Beta-style calibration. Little ECE/Brier remains for richer post-hoc maps to recover at K=5. Isotonic or binning would probably hurt Brier and robustness to shift.

### Cited findings
- Temperature scaling (TS) uses one parameter, T. It leaves the argmax, and so accuracy, unchanged. Guo et al. found it the most effective method on most vision and NLP benchmarks. Vector and matrix scaling overfit when there are many classes, and matrix scaling did badly on CIFAR-100. ECE was computed with 15 equal-width bins. [K] — [Guo et al. 2017, ICML](https://arxiv.org/abs/1706.04599)
- Fine-tuned transformers are already fairly well calibrated in-domain. Pre-trained models (BERT/RoBERTa) had lower ECE than non-pre-trained baselines on QQP and SWAG, and RoBERTa was always better calibrated in-domain than BERT. After TS, RoBERTa in-domain ECE was "0.7–0.8". The learned T was 1.0–1.4 in-domain (BERT 1.20/1.34/0.99; RoBERTa 1.16/1.39/1.10 for SNLI/QQP/SWAG), and ECE used k=10 bins. Conclusion: "MLE training yields scores that are fundamentally good but just need some minor rescaling." [V] — [Desai & Durrett 2020, EMNLP](https://arxiv.org/abs/2003.07892). (The PDF table extracted garbled, so individual out-of-the-box ECE cells are omitted. Check Table 3 in the paper.)
- Dirichlet calibration is a multinomial generalisation of Beta calibration: matrix scaling applied to log-probabilities, with off-diagonal and intercept regularisation (ODIR). The authors report it improves classwise-ECE and log-loss over TS mainly when there are many classes, and is competitive otherwise. Code: `dirichletcal`. [K] — [Kull et al. 2019, NeurIPS](https://arxiv.org/abs/1910.12656)
- Beta calibration (binary) is p' = σ(a·ln p − b·ln(1−p) + c). It includes the identity map, and Platt scaling on log-odds is the special case a = b. [K] — [Kull, Silva Filho & Flach 2017, AISTATS](https://proceedings.mlr.press/v54/kull17a.html)
- Mix-n-Match: TS is data-efficient and keeps accuracy but is not very expressive. Isotonic is expressive but data-inefficient, and "can degrade accuracy, particularly in small-data regimes". Ensemble Temperature Scaling (ETS) is a convex mix of a TS-scaled softmax, the raw softmax and the uniform distribution, and keeps accuracy. The authors also say histogram ECE "may provide misleading results especially in small-data regime" and propose a KDE-based estimator instead. [V, from abstract/summary] — [Zhang, Kailkhura & Han 2020, ICML](https://arxiv.org/abs/2003.07329)
- Parameterized Temperature Scaling (PTS) learns a temperature for each input with a small neural net, trading some data efficiency for more expressive power. It is reported to beat TS on vision benchmarks. [K] — [Tomani et al. 2022, ECCV](https://arxiv.org/abs/2102.12182)
- Spline calibration fits a spline to the cumulative distribution of scores and evaluates with a KS-style calibration error that needs no binning. [K] — [Gupta et al. 2021, ICLR](https://arxiv.org/abs/2006.12800)
- Scaling methods (Platt/TS) are "less calibrated than reported": plug-in binned ECE underestimates their error. Histogram binning has measurable calibration error but needs far more samples, so the authors propose a scaling-binning calibrator. [K] — [Kumar, Liang & Ma 2019, NeurIPS "Verified Uncertainty Calibration"](https://arxiv.org/abs/1909.10155)
- Top-label calibration is a weaker and more practical target than canonical multiclass calibration. Gupta & Ramdas give binning-based top-label calibrators that come with guarantees. [K] — [Gupta & Ramdas 2022, ICLR](https://arxiv.org/abs/2107.08353)
- Classwise ECE and adaptive (equal-mass) binning are recommended over top-label ECE with equal-width bins. [K] — [Nixon et al. 2019, "Measuring Calibration in Deep Learning"](https://arxiv.org/abs/1904.01685)
- Temperature scaling breaks down when class supports overlap. For ERM-trained models, TS gets worse as overlap grows and approaches chance-level calibration with many classes. Added label noise, which raises overlap, causes the same degradation on image benchmarks. A modified Mixup training objective calibrates reasonably well despite overlap. [V, abstract] — [Chidambaram & Ge 2024, ICLR](https://arxiv.org/abs/2306.00740)
- Penso's thesis reportedly finds TS "highly susceptible to label noise" in the calibration set, sometimes ending worse than the uncalibrated model. [V only via search snippet; not read in full] — [Penso, "Confidence Calibration of Deep Learning Systems" (2026)](https://arxiv.org/pdf/2608.12100)
- Library support: `netcal` (TS, vector/matrix scaling, Beta, histogram binning, BBQ, ENIR); `dirichletcal`; `relplot` (smECE); `crepes` and `MAPIE` (conformal). [K] — [netcal docs](https://efs-opensource.github.io/calibration-framework/), [relplot](https://pypi.org/project/relplot/)

### Inferences
- **Route (K=5)**: (1−ε)·softmax(z/T + b) + ε/5 is essentially ETS with a bias term. That is the recommended family for 1.5k–6.5k samples. Dirichlet/matrix scaling has K²+K = 30 parameters, which 6.5k OOF points can support with ODIR. The only plausible gain is from fixing systematic confusion between particular class pairs, so expect at most a few hundredths of ECE. It is worth one experiment, with an L2 pull toward the identity matrix, scored by Brier on held-out folds.
- **Binaries**: lo + (hi−lo)·σ(a·z + b) adds floor and ceiling parameters to Platt. Beta calibration adds the asymmetric (a ≠ b) term on log p vs log(1−p), which helps when the score distribution is skewed. Fitting Beta inside the floor/ceiling wrapper costs one extra parameter.
- **Avoid isotonic regression and histogram binning as the final map.** They produce step functions. Step functions can lower binned ECE on the very data they were fit on, but they raise Brier through lost refinement and ties, and they do not extrapolate under shift. Smooth parametric maps degrade more gracefully.
- An input-dependent temperature (PTS-style, or temperature as a function of text length or "vagueness") is the only post-hoc family that addresses Chidambaram & Ge's overlap problem. Here the overlap is vague tickets labelled near-uniformly. A single global T has to compromise between confident, clear tickets and vague ones.

### Gaps
- I found no head-to-head benchmark of TS vs ETS vs Dirichlet vs Beta on text classifiers with 1.5k–6.5k calibration examples and K≈5. Recommendations here are extrapolated from vision benchmarks and from theory.
- Mix-n-Match did not report a direct comparison against Dirichlet calibration in the material retrieved.

## 2. Training-time calibration for fine-tuned transformers: label smoothing, focal loss, mixup, MbLS, ensembles, MC dropout, SWAG, Laplace

### Takeaway
For fine-tuned encoders, MLE followed by post-hoc TS is the best in-domain recipe. Label smoothing hurts in-domain calibration after TS, helps under strong shift, and damages the confidence *ranking* that selective prediction (abstention) relies on. Ensembles and other model averaging are the most consistently robust option under shift. Choosing the checkpoint by refinement (post-TS loss) rather than raw validation loss is a cheap, well-founded improvement.

### Cited findings
- In-domain, MLE beats label smoothing (LS, α=0.1): "temperature scaling hurts LS in many cases". Out of domain, LS is "generally more effective". MLE models had out-of-domain ECE of 8–12, and on HellaSWAG RoBERTa-LS had 5.8× lower ECE than RoBERTa-MLE. The temperature needed out of domain was much larger (2.79–3.61 for TwitterPPDB/HellaSWAG vs 1.0–1.4 in-domain), "suggest[ing] the degree of distribution shift and magnitude of T may be closely related." [V] — [Desai & Durrett 2020](https://arxiv.org/abs/2003.07892)
- Label smoothing degrades selective classification. LS suppresses the max logit more on samples likely to be correct than on likely errors. A larger α makes selective classification worse. On ImageNet at 1% risk, CE models could accept about 15.66% of samples and LS models about 0%. Normalising logits post hoc (dividing by their p-norm before softmax) recovers the loss. [V via secondary summaries; vision experiments] — [Xia et al. 2025, ICLR, "Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It"](https://arxiv.org/abs/2403.14715)
- Popular calibration methods (LS, focal loss, mixup, TS-style training) often worsen the separation in confidence between correct and wrong predictions, which hurts failure prediction. Flat-minima methods (SWA, SAM) help instead. [V, abstract] — [Zhu et al. 2022, ECCV, "Rethinking Confidence Calibration for Failure Prediction"](https://arxiv.org/abs/2303.02970)
- Label smoothing erases information in the logits about class similarity, and it hurts knowledge distillation. [K] — [Müller, Kornblith & Hinton 2019](https://arxiv.org/abs/1906.02629)
- Label smoothing can mitigate symmetric label noise, acting like loss correction. [K] — [Lukasik et al. 2020, ICML](https://arxiv.org/abs/2003.02819)
- Focal loss improves calibration, and the sample-dependent γ variant (FLSD-53) beats CE+TS on several vision tasks. [K] — [Mukhoti et al. 2020, NeurIPS](https://arxiv.org/abs/2002.09437)
- Margin-based label smoothing (MbLS) puts an inequality penalty on logit distances, and outperforms LS and focal loss. [K] — [Liu et al. 2022, CVPR](https://arxiv.org/abs/2111.15430)
- Mixup-trained networks are better calibrated and less overconfident on OOD data. [K] — [Thulasidasan et al. 2019, NeurIPS](https://arxiv.org/abs/1905.11001)
- For PLM text classification, Kong et al. add on-manifold and off-manifold smoothness regularisers (mixup-like) during fine-tuning and report lower in-domain and OOD ECE across six datasets. [K] — [Kong et al. 2020, EMNLP](https://arxiv.org/abs/2010.11506)
- "Bag of Tricks for In-Distribution Calibration of Pretrained Transformers": ensembles that overfit the training set calibrate poorly, and confidence-penalty losses trade calibration against accuracy. The combined recipe CALL (confidence penalty + MixUp + ensembling) improves both, especially in low-resource settings. [V, abstract; numbers not retrieved] — [Kim et al. 2023, Findings of EACL](https://arxiv.org/abs/2302.06690)
- PLMs do not learn to be calibrated during fine-tuning: confidence keeps rising with training whether or not predictions are correct. Learnable calibration methods help. [K] — [Chen et al. 2023, ACL, "A Close Look into the Calibration of Pre-trained Language Models"](https://arxiv.org/abs/2211.00151)
- Refine, then calibrate: calibration error and refinement error reach their minima at different epochs, so early stopping on validation loss is "suboptimal for both". The proposal is to select epochs and hyperparameters by refinement error (e.g., the loss after TS) and then calibrate post hoc. [V, abstract] — [Berta, Holzmüller, Jordan & Bach 2025](https://arxiv.org/abs/2501.19195)
- Deep ensembles (M≈5) give strong calibration and OOD uncertainty. [K] — [Lakshminarayanan et al. 2017](https://arxiv.org/abs/1612.01474)
- Under dataset shift, "traditional post-hoc calibration does indeed fall short", while "methods that marginalize over models give surprisingly strong results across a broad spectrum of tasks". Deep ensembles were the best performer overall. [V, abstract] — [Ovadia et al. 2019, NeurIPS](https://arxiv.org/abs/1906.02530)
- MC dropout [K] — [Gal & Ghahramani 2016](https://arxiv.org/abs/1506.02142). SWAG [K] — [Maddox et al. 2019](https://arxiv.org/abs/1902.02476). Last-layer Laplace is cheap and reduces overconfidence far from the data [K] — [Kristiadi et al. 2020](https://arxiv.org/abs/2002.10118); [Daxberger et al. 2021, "Laplace Redux"](https://arxiv.org/abs/2106.14806). Laplace on LoRA parameters improves calibration of fine-tuned LLMs [K] — [Yang et al. 2024, ICLR](https://arxiv.org/abs/2308.13111).
- Proper scoring rules (log-loss, Brier) decompose into calibration plus refinement. Calibration methods should be judged on proper scores, not ECE alone. [K] — [Gruber & Buettner 2022, NeurIPS](https://arxiv.org/abs/2203.07835)

### Inferences
- **Do not add label smoothing to training.** Abstention is 15% of the score, and LS degrades selective prediction. Decide's post-hoc ε-floor captures the calibration benefit of smoothing without touching the logit ranking. It is a monotone transform of each class probability, so for route the argmax and the ordering by max-probability are unchanged and risk@coverage is preserved.
- **Cheapest strong gain: ensemble the k fold models.** Decide already has 6,500 OOF predictions, which implies k-fold training. Averaging the k fold models' logits or probabilities at test time is a deep-ensemble-lite, and Ovadia's evidence favours it under shift. Cost is k× CPU inference on a 17M model, roughly 5 × a few ms per ticket. Recalibrate on the ensemble's own OOF-like predictions, because ensembling changes the effective temperature.
- **Select the checkpoint by post-TS validation loss** (Berta et al.), not by raw validation loss or accuracy.
- Last-layer Laplace on the 17M encoder head is cheap on CPU. Its main benefit is softer confidence far from the training data, which matters under judge shift. Expected gains are small in-domain once a TS floor is in place.
- Mixup or manifold regularisation is the training-time remedy that Chidambaram & Ge and Kong et al. point to for overlapping classes (vague tickets). It costs retraining, and its effect on selective prediction is uncertain (Zhu et al. 2022 found mixup can hurt failure prediction).

### Gaps
- I found no study of these methods on encoders as small as Ettin-17M. All transformer evidence is from BERT/RoBERTa-base/large.
- The detailed numbers for Kim et al. 2023 and Kong et al. 2020 were not retrieved.
- Wang et al. NeurIPS 2021 ("Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence"), which reports that LS, mixup and focal loss make post-hoc calibration harder, could not be verified in-session.

## 3. Calibration under distribution shift (judges may use a different dataset)

### Takeaway
Every post-hoc map fit on in-distribution data loses calibration under shift, usually toward **overconfidence**, because shift lowers accuracy more than it lowers confidence. The defences with evidence behind them are model averaging (ensembles), maps with few parameters, maps that are conservative by construction (floors, ceilings, larger T), calibrating on synthetically perturbed or multi-domain data, and, given unlabeled target data, importance-weighted calibration (TransCal). Bounded maps like Decide's degrade gracefully by construction: worst-case confidence is capped.

### Cited findings
- Post-hoc calibration fit on i.i.d. validation data "fall[s] short" under shift, and model-marginalising methods (ensembles) hold up best. [V] — [Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)
- The temperature needed grows with the severity of shift (in-domain T 1.0–1.4 vs OOD 2.8–3.6 for RoBERTa/BERT). LS models are better calibrated out of the box under strong shift. [V] — [Desai & Durrett 2020](https://arxiv.org/abs/2003.07892)
- Calibration under shift varies by architecture. Modern non-convolutional models are among the best calibrated in and out of distribution, and in-distribution calibration does not perfectly predict calibration under shift. [K] — [Minderer et al. 2021, NeurIPS, "Revisiting the Calibration of Modern Neural Networks"](https://arxiv.org/abs/2106.07998)
- Multi-domain temperature scaling (MDTS): TS fit on pooled data can fail on individual domains "even under mild subpopulation shift". MDTS learns a temperature per input by regressing domain temperatures on features, and improves both in-distribution and OOD calibration on three benchmarks. [V, abstract] — [Yu, Bates, Ma & Jordan 2022, NeurIPS](https://arxiv.org/abs/2206.02757)
- Post-hoc calibration for domain drift: fit TS or ETS on a validation set perturbed at a range of noise strengths, so the calibrator sees a spectrum of shifts. This improves calibration under shift while matching TS in-domain. [K] — [Tomani et al. 2021, CVPR](https://arxiv.org/abs/2012.10988)
- TransCal: domain-adaptation models gain accuracy "at the expense of well-calibrated probabilities". TransCal is a hyperparameter-free post-hoc method that uses importance-weighted target calibration with lower bias and variance, and it lowers calibration error for existing DA methods. It needs unlabeled target data. [V] — [Wang, Long, Wang & Jordan 2020, NeurIPS](https://arxiv.org/abs/2007.08259)
- Consistency-guided TS for out-of-domain calibration uses style/content augmentations of the calibration set, and compares against MDTS. [V, metadata only] — [Lin et al. 2024, AAAI](https://arxiv.org/abs/2402.15019)
- Calibration under domain generalisation: train a calibrator across several source domains (or clusters) so it transfers to unseen domains. [K, title only; arXiv ID not verified] — Gong et al. 2021, ICCV, "Confidence Calibration for Domain Generalization under Covariate Shift".
- Transferable calibration for PLMs across domains is also studied at AAAI (2023). [V, title only] — [Transferable Post-hoc Calibration on Pretrained Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/26632/26404)

### Inferences
- **Bounded maps cap worst-case overconfidence.** With ε≈0.06 the route floor caps max confidence at 1 − 0.8ε ≈ 0.952. If accuracy drops under shift, ECE is still bounded by (cap − accuracy) instead of (1 − accuracy). Keep the floor and ceiling even if they cost about 0.001 in-domain.
- **Shift-hedging option: fit T (and ε) on a mixture of the dev set and perturbed dev text** (Tomani-style). Perturbations could be paraphrase, typo noise, truncation, or tickets from another domain. This biases toward slightly higher T, which costs a little in-domain and buys robustness. Decide is scored on an unknown dataset, so a small deliberate under-confidence is a reasonable hedge. Over-confidence is typically what shift produces.
- **If judges supply unlabeled test inputs at inference time** (batch scoring), test-time options include: (a) TransCal or importance-weighted TS using a domain classifier; (b) label-shift re-estimation via EM (Saerens et al. 2002 — not retrieved this session) to update the route class prior/bias b. Both are risky without labels. Use only with shrinkage toward the fitted parameters.
- Structural parameters fit on OOF data from *your* distribution (for example ε = the route noise rate, the approve ceiling ≈ 0.8) transfer only if the judge data has the same noise process. If judge labels are cleaner, ε is too high and the model is under-confident. If noisier, the reverse. A moderate ε hedges both ways.

### Gaps
- I found no direct evidence on how Platt or TS fit on one text domain transfers to another for small encoders. Desai & Durrett's OOD T values were fit on OOD data, which shows what the required T is, not that an in-domain T transfers.
- Gong et al. 2021 and Park et al. 2020 ("Calibrated prediction with covariate shift via unsupervised domain adaptation", AISTATS) were not verified in-session.

## 4. Calibration under label noise: noisy vs clean calibration targets when test labels are equally noisy

### Takeaway
If the test labels come from the same noise process as the calibration labels, **calibrating to the noisy-label distribution is exactly the right target**. ECE and Brier measured on noisy test labels reward predicting the noisy posterior p̃(y|x), and log-loss and Brier are proper scoring rules for whatever the label distribution is. For uniform random flips at rate ε, the noisy posterior is (1−ε)·p_clean + ε/K. That is Decide's route map, and it is equivalent to forward loss correction with a symmetric transition matrix. Clean-label correction (T-revision, noise-corrected ECE) matters only if the *evaluation* labels are clean.

### Cited findings
- Strictly proper scoring rules (log score, Brier) are minimised in expectation only by reporting the true conditional distribution of the observed outcome. [K] — [Gneiting & Raftery 2007, JASA](https://doi.org/10.1198/016214506000001437)
- Forward loss correction: under a transition matrix T with p(ỹ|x) = Tᵀ p(y|x), training on the noisy labels through T recovers the clean posterior, and the model's output passed through T predicts the noisy label. [K] — [Patrini et al. 2017, CVPR](https://arxiv.org/abs/1609.03683). T-revision estimates T without anchor points. [K] — [Xia et al. 2019, NeurIPS](https://arxiv.org/abs/1906.00189)
- Treating noisy labels as ground truth "can bias calibration evaluation" relative to the *clean* distribution. Hanada & Futami propose Noise-Corrected ECE (NC-ECE) and noise-corrected recalibration. Uncorrected recalibration on noisy annotations "can degrade" calibration with respect to the latent clean labels, and NC-ECE "closely tracks its clean-label oracle counterpart" on semi-synthetic benchmarks. [V, abstract; no numbers on page] — [Hanada & Futami, Oct 2026](https://arxiv.org/abs/2610.19543)
- Noisy test labels bias error estimates. Even 1% wrong test labels make a classifier with a true 6% error appear to have 6.88% error. [V, via search summary] — [Lam & Stork 2003, IJCAI](https://www.ijcai.org/Proceedings/03/Papers/076.pdf)
- LWCal is a loss-weighted isotonic calibrator for a noisy calibration set evaluated on clean test labels, for tabular classifiers. [V, metadata] — [LWCal 2026](https://arxiv.org/html/2609.26839)
- Temperature scaling degrades with label noise because noise increases class overlap. [V] — [Chidambaram & Ge 2024](https://arxiv.org/abs/2306.00740)
- Label errors are pervasive in benchmark *test* sets (on average about 3.3% across 10 datasets). Rankings between models can change when the test sets are corrected. [K] — [Northcutt, Athalye & Mueller 2021, NeurIPS D&B](https://arxiv.org/abs/2103.14749)

### Inferences
- **Route**: random flips are label-independent of the text, so p̃(y|x) = (1−ε)p(y|x) + ε/5 exactly. The ε-floor on top of TS is the Bayes-correct form. A *learned* T alone cannot represent it, because TS cannot produce a floor. This explains the +0.65 points over plain TS. Fitting ε on OOF data from the same pipeline is correct. ε should approximate the actual flip rate times (5/4 or 1, depending on whether a "flip" can land on the true class).
- **Vague tickets labelled near-uniformly** are not noise: their true p(y|x) is near uniform. The model should *predict* near-uniform for them, and that requires local (input-dependent) softening. The global floor cannot provide it. Candidates: input-dependent temperature (PTS/MDTS-style) with a "vagueness" feature, or training with these items kept in (see §6).
- **Approve**: refusals driven by hidden account history are aleatoric. The Bayes-optimal p(approve|text) is capped at roughly the base rate of non-hidden-refusal, about 0.8, for texts that look approvable. The learned ceiling "hi" in lo + (hi−lo)σ(·) encodes this correctly. Forcing probabilities toward 1 would be penalised by Brier on a noisy test.
- **If the judge's labels are cleaner** (for example, a hand-relabelled set), the ε-floor and ceiling make Decide under-confident. On Brier, the cost is bounded and small: for ε=0.06 it is on the order of ε² per sample on correct confident items. On ECE, the cost is about 0.8ε ≈ 0.05 on confident bins, which is large. This trade-off is a judgement call. A smaller ε than the OOF estimate hedges.

### Gaps
- I found no paper that directly compares fitting calibration to noisy labels vs clean labels when the test set is equally noisy. The answer above follows from proper-scoring-rule theory and the forward-correction identity, not from a dedicated empirical study.
- The precise ε vs flip-rate relationship depends on how the hackathon generated the flips (uniform over all 5 classes or over the 4 others), which is not known here.

## 5. Conformal prediction: split conformal, APS/RAPS, Mondrian, conformal risk control, conformal under shift, and abstention at 80% coverage

### Takeaway
Conformal prediction gives finite-sample *marginal coverage* guarantees for prediction sets, and with dispersive label noise those guarantees hold conservatively. It does **not** improve the ranking quality that determines risk at a fixed 80% coverage. Abstaining by set size is close to thresholding the same score. Its useful roles for Decide: (a) setting the abstention threshold with a statistical guarantee (Learn-then-Test, conformal risk control), and (b) per-class (Mondrian) thresholds. Under shift, the guarantees fail unless weighted conformal or online adaptive conformal is used.

### Cited findings
- Gentle introduction to split conformal, APS and RAPS, conformal risk control and conformal under shift. [K] — [Angelopoulos & Bates 2021/2023](https://arxiv.org/abs/2107.07511)
- APS (adaptive prediction sets) is an adaptive cumulative-probability score [K] — [Romano, Sesia & Candès 2020, NeurIPS](https://arxiv.org/abs/2006.02544). RAPS adds regularisation for smaller, more stable sets [K] — [Angelopoulos et al. 2021, ICLR](https://arxiv.org/abs/2009.14193).
- Conformal risk control extends coverage guarantees to any bounded monotone loss. [K] — [Angelopoulos et al. 2022/2024, ICLR](https://arxiv.org/abs/2208.02814). Learn-then-Test picks thresholds (e.g., for abstention) with distribution-free risk guarantees. [K] — [Angelopoulos et al. 2021](https://arxiv.org/abs/2110.01052)
- Class-conditional and clustered conformal give coverage per class, so rare classes are covered as well as common ones. [K] — [Ding et al. 2023, NeurIPS](https://arxiv.org/abs/2306.09335)
- Under covariate shift, weighted conformal reweights calibration scores by the likelihood ratio [K] — [Tibshirani et al. 2019](https://arxiv.org/abs/1904.06019). Beyond exchangeability, the coverage gap is bounded by total-variation drift [K] — [Barber et al. 2023](https://arxiv.org/abs/2202.13415). Adaptive conformal inference handles online shift [K] — [Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170).
- Label noise: conformal prediction and risk control "attain conservative risk over the clean ground truth labels whenever the noise is dispersive and increases variability". Adversarial noise can make sets too small. [V] — [Einbinder et al. 2024, JMLR](https://jmlr.org/papers/v25/23-1549.html); [Feldman et al. 2023](https://proceedings.mlr.press/v204/feldman23a/feldman23a.pdf)
- Selective conformal risk control: first decide which inputs to accept, then build conformal sets only for accepted inputs. On CIFAR-10 and diabetic retinopathy this gave much smaller sets for accepted samples with risk still controlled. The choice of selection score (MSP, margin, entropy) changes efficiency but not validity. [V, via summary] — [Selective Conformal Risk Control, 2025](https://arxiv.org/abs/2512.12844)
- Treat sets above a size cap as "uninformative → defer". This unifies explicit and implicit abstention under a size-based rule. [V, via search summary] — [Uncertainty-Driven Reliability survey 2025](https://arxiv.org/pdf/2508.07556)
- Selective classification (risk–coverage, softmax-response thresholding) with guaranteed risk. [K] — [Geifman & El-Yaniv 2017, NeurIPS](https://arxiv.org/abs/1705.08500)

### Inferences
- **For risk@80% coverage, what matters is the ordering of confidence scores** (AURC-style), not calibration or set validity. For a binary task, conformal set = {both labels} ⇔ max p < threshold, so it is the same ranking. For 5-way route, APS set size is monotone in the cumulative mass of the top classes. It can rank slightly differently from max-softmax (margin or entropy-like), which might help a little. Test MSP vs margin vs APS-size as abstention scores on OOF data.
- **Cost-sensitive approve (3:1)**: the Bayes decision is to pick the action with lower expected cost and abstain when min expected cost exceeds the abstention cost (Chow's rule). This needs *calibrated* p, which ties abstention to calibration. Conformal adds a finite-sample bound on the error of the accepted set, which is useful if the scorer penalises exceeding a risk target.
- Conformal does not change p itself, so it gives **no direct ECE/Brier gain**.
- Under judge shift, split-conformal thresholds lose their guarantee. A fixed *coverage* target (80%) is better met by a quantile of the scores on the test batch itself, if batch inference is allowed. That self-adjusts to shift (an inference; it is not a conformal guarantee).

### Gaps
- I found no head-to-head risk–coverage comparison between softmax-response abstention and conformal-set-size abstention on text classifiers.
- Sesia et al. 2024 (adaptive conformal classification with noisy labels) was not retrieved.

## 6. Noisy-label learning for pretrained text encoders: co-teaching, DivideMix, GCE, SCE, ELR, cleanlab, loss truncation

### Takeaway
Pretrained encoders such as BERT are already robust to *random* label noise at low rates. Noise-handling methods "do not always improve its performance, and may even deteriorate it" (Zhu et al. 2022). Feature-dependent noise is harder, and gains there are mixed. At about 5–6% random flips with equally noisy test labels, cleaning or robust losses cannot raise the noisy-test ceiling. Removing the uniformly labelled vague tickets would likely *hurt* calibration, since they teach the model when to be uncertain.

### Cited findings
- BERT is robust to injected random label noise. Existing noise-handling methods (noise transition matrix, co-teaching, label smoothing) "do not always improve its performance, and may even deteriorate it". Under feature-dependent (weak-supervision) noise, none of the methods improved BERT's peak performance, and early stopping on a clean validation set matters. [V] — [Zhu, Hedderich, Zhai, Adelani & Klakow 2022, Insights from Negative Results in NLP](https://aclanthology.org/2022.insights-1.8/) ([arXiv](https://arxiv.org/abs/2204.09371))
- With realistic noise, some methods help for feature-dependent noise. For synthetic noise from multiple annotations, the authors "do not observe significant gains". [V, via summary] — [Agro & Aldarmaki 2023](https://arxiv.org/abs/2305.16337)
- BERT's performance can drop by up to 35% under 50% weak-supervision noise (Hausa), i.e., at high, structured noise. [V, via summary] — [How Much Noise Can BERT Handle? (2026)](https://arxiv.org/pdf/2603.07346)
- PLM-based label error detection: rank examples by the fine-tuned PLM's loss on its own label. This beats more complex methods at finding real label errors in NLP datasets. [K] — [Chong, Hong & Lipton 2022, EMNLP](https://arxiv.org/abs/2205.12702)
- Confident learning (cleanlab) estimates the joint distribution of noisy and true labels from out-of-sample probabilities, then prunes likely errors. [K] — [Northcutt, Jiang & Chuang 2021, JAIR](https://arxiv.org/abs/1911.00068)
- Co-teaching: two networks exchange small-loss samples [K] — [Han et al. 2018](https://arxiv.org/abs/1804.06872). DivideMix: GMM split into clean and noisy sets plus semi-supervised MixMatch [K] — [Li, Socher & Hoi 2020, ICLR](https://arxiv.org/abs/2002.07394). GCE [K] — [Zhang & Sabuncu 2018](https://arxiv.org/abs/1805.07836). SCE [K] — [Wang et al. 2019, ICCV](https://arxiv.org/abs/1908.06112). ELR (early-learning regularisation) [K] — [Liu et al. 2020, NeurIPS](https://arxiv.org/abs/2007.00151). Loss truncation drops high-loss examples [K] — [Kang & Hashimoto 2020, ACL](https://arxiv.org/abs/2004.14589). These were mostly validated on vision with 20–80% synthetic noise.
- Networks learn clean patterns first and memorise noisy labels later, which is why early stopping is an effective defence. [K] — [Arpit et al. 2017](https://arxiv.org/abs/1706.05394)

### Inferences
- **The expected gain is small.** At ε≈5–6% random flips, an encoder fine-tuned with early stopping is near its noise-robust optimum (Zhu et al. 2022). The noisy-test accuracy ceiling is about (1−ε)·clean_acc + ε/5, whatever the training method.
- **Cleaning helps the 40% quality component only slightly,** if at all. A model trained on cleaned labels predicts the clean label more sharply, but the test labels still contain flips, so measured accuracy improves only by the share of currently "memorised" errors. Re-apply the ε-floor afterwards for calibration.
- **Do not prune vague, uniformly labelled tickets.** Cleanlab and loss truncation would flag them as label errors (high loss, low self-confidence). Removing them makes the model confident on vague text, which raises ECE and Brier and hurts abstention. If cleaning, prune only examples confidently assigned to a *different* single class. A safer alternative is to down-weight them rather than drop them.
- **The approve task has no "label error" to clean.** Hidden-history refusals are genuine aleatoric outcomes. Robust losses like GCE/SCE, which under-fit hard examples, would push p(approve) toward 1 on refused-but-approvable-looking text and break calibration.
- **Worth trying (cheap):** ELR or early stopping on post-TS loss; label-error detection from the existing OOF losses (Chong et al.) to *inspect* rather than delete data.

### Gaps
- I found no evidence specific to Ettin or ModernBERT-style small encoders.
- I found no study that measures the effect of noisy-label methods on *calibration* (rather than accuracy) for PLM text classifiers.

## 7. ECE pitfalls: binning, estimators (debiased ECE, smECE), and the interaction with Brier skill

### Takeaway
Binned ECE is a biased and noisy estimator. Equal-width bins overestimate ECE for well-calibrated models, the bias grows as n shrinks, and results depend on the number of bins. Equal-mass, debiased, sweep and smooth (kernel) estimators are better. Since the judges' ECE implementation is unknown, the safest strategy is to minimise proper scores (log-loss/Brier) with smooth, monotone maps, and then check that ECE is low under several estimators. Brier skill score depends on the reference (base-rate) forecast, so it moves with class priors under shift.

### Cited findings
- Equal-mass bins have lower bias than equal-width. Recommended estimators are debiased equal-mass ECE (Kumar et al.) and ECE_sweep (as many bins as possible while the calibration function stays monotone). Equal-width ECE_bin is strongly biased for perfectly calibrated models. [V] — [Roelofs et al. 2022, AISTATS](https://arxiv.org/abs/2012.08668)
- Debiased plug-in ECE estimator, and the finding that binned ECE underestimates the true calibration error of scaling methods. [K] — [Kumar et al. 2019](https://arxiv.org/abs/1909.10155)
- SmoothECE: kernel-smooth the residuals with an RBF kernel, then compute ECE. With a principled automatic bandwidth it is a consistent calibration measure; implemented in `relplot`. [V] — [Błasiok & Nakkiran 2024, ICLR](https://arxiv.org/abs/2309.12236); [relplot](https://pypi.org/project/relplot/)
- Theory of "distance to calibration", with binned ECE shown to be discontinuous and not a consistent calibration measure. [K] — [Błasiok, Gopalan, Hu & Nakkiran 2023, STOC](https://arxiv.org/abs/2211.16886)
- Histogram ECE can mislead with small data, and a KDE-based ECE is proposed. [V] — [Zhang et al. 2020](https://arxiv.org/abs/2003.07329)
- Proper scores should be the primary target, because ECE can be zero for an uninformative constant predictor. [K] — [Gruber & Buettner 2022](https://arxiv.org/abs/2203.07835)

### Inferences
- With about 1.5k dev items and 10–15 bins, each bin holds about 100–150 items. The sampling SE of bin accuracy is roughly √(0.25/100) ≈ 0.05 in mid-confidence bins, so a perfectly calibrated model shows a positive ECE of about 0.01–0.02 from noise alone. Differences between calibrators below about 0.005 ECE on dev are not distinguishable. Use the 6.5k OOF set with bootstrap CIs to choose methods.
- **Brier skill score** = 1 − Brier/Brier_ref, where the reference is the climatology forecast. Under prior shift, Brier_ref changes, so BSS can rise or fall without any change in the model. Errors in the route bias term b (class priors) affect both ECE and BSS. Shrink b toward 0 for robustness.
- Methods that produce many tied probabilities (histogram binning, isotonic) can game equal-width binned ECE but lose resolution, which hurts Brier and therefore BSS. This is another argument against them for a combined ECE+BSS score.
- Report and optimise for several ECE variants: 15-bin equal-width, equal-mass, and smECE. Robustness across estimators is a proxy for robustness to whatever implementation the judges use.

### Gaps
- The exact ECE definition the judges use (top-label vs classwise, number of bins, equal-width vs equal-mass) is unknown. Whether route and binary ECE are averaged or pooled is also unknown.
- I found no published guidance on combining ECE and BSS into a single score, or on which method optimises that combination.
