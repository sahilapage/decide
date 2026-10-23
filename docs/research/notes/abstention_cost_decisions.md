# Selective prediction, abstention, learning-to-defer and cost-sensitive decisions for a small text classifier (Decide)

Scope note: "fetched" = I read the primary source (abstract/PDF text) this session. Items marked [link-only] are cited for existence and method definition from well-known arXiv IDs. I did not re-fetch them, so their numbers are not reported here. Current as of 2026-10-24.

Our metric, restated: AbstScore = 1 − risk@80%cov / risk@100%cov, with cases ranked by max(p, 1−p) (binary) or top-class probability (route). Random ranking gives ≈0. A perfect ranking gives 1 if the full-coverage error is ≤20%. Our 0.667 means the retained 80% has one third of the full-coverage error rate.

## 1. Which confidence scores beat MSP for selective classification on calibrated, fine-tuned transformers, especially when errors are aleatoric?

### Takeaway
MSP / softmax response (SR) is still the baseline that most methods fail to beat consistently in large benchmarks. That holds in vision (Jaeger et al. ICLR 2023) and across 17 NLP datasets (Varshney et al. 2022). Learned selection heads (SelectiveNet, Deep Gamblers, SAT) do not beat SR as scoring functions: Feng et al. 2023 attribute their gains to a better-trained backbone. Density and distance scores (Mahalanobis, DDU, SNGP, kNN) help mainly where errors are epistemic (low-resource, OOD, covariate shift). On ambiguous, aleatoric in-distribution text tasks they often lose to SR. The only reliable wins come from combining SR with a density score (HUQ, kNN-UE), and those wins are small (single-digit to ~13% relative AURC reduction).

### Cited Findings
- **MSP baseline.** Hendrycks & Gimpel (ICLR 2017) proposed maximum softmax probability as the baseline for misclassification and OOD detection. [link-only] — [arXiv 1610.02136](https://arxiv.org/abs/1610.02136)
- **Softmax response for selection.** Geifman & El-Yaniv (NeurIPS 2017) built selective classifiers on a trained net using softmax response (SR) and MC-dropout as confidence-rate functions. Their SGR algorithm binary-searches a threshold that guarantees a user-set risk with high probability. Headline result: 2% top-5 ImageNet error guaranteed with probability 99.9% at almost 60% coverage. — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
- **Jaeger et al., ICLR 2023 (oral).** "A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification" benchmarks confidence scoring functions across failure sources (in-distribution misclassification and several shift types). It reports "the revelation of a simple softmax response baseline as the overall best performing method". It argues AURC is the right holistic metric. — [arXiv 2211.15259](https://arxiv.org/abs/2211.15259)
- **Varshney, Mishra & Baral, Findings ACL 2022.** Across 17 NLP datasets in IID, OOD and adversarial settings, "none of the existing approaches consistently and considerably outperforms MaxProb", even with extra held-out data or compute. Method rankings do not transfer between tasks: MC-Dropout was best on duplicate detection but weak on NLI, especially OOD. The metric was area under the risk-coverage curve. — [arXiv 2203.00211](https://arxiv.org/pdf/2203.00211)
- **Xin et al., ACL 2021 ("The Art of Abstention").** Pretrained transformers improve both accuracy and confidence estimation. MC-dropout "performs consistently worse than softmax response" on NLP tasks and needs at least 20 stochastic passes to match SR. Their "error regularization" adds a pairwise ranking penalty during fine-tuning: if example i has higher error than example j, i's MSP should not exceed j's. It improves selective AUC and RPP without hurting accuracy. Code: castorini/transformers-selective. — [ACL Anthology 2021.acl-long.84](https://aclanthology.org/2021.acl-long.84.pdf)
- **Feng et al., ICLR 2023 ("Towards Better Selective Classification").** Selective-head methods (SelectiveNet's selection head, Deep Gamblers' or SAT's extra abstention logit) gain because they train a more generalizable classifier, not because of the head. Training with those methods and then selecting by the classifier's softmax response "consistently" beats the head across coverages and datasets. Adding an entropy-minimization regularizer gives the new SOTA. The paper is vision-focused. — [arXiv 2206.09034](https://arxiv.org/abs/2206.09034); [ICLR 2023 poster](https://iclr.cc/virtual/2023/poster/11355)
- **Learned-confidence methods (definitions).** [link-only]
  - SelectiveNet: jointly trained selection head with a coverage constraint. — [arXiv 1901.09192](https://arxiv.org/abs/1901.09192)
  - Deep Gamblers: (C+1)-th abstention output, trained with a portfolio/gambling loss. — [arXiv 1907.00208](https://arxiv.org/abs/1907.00208)
  - ConfidNet: regresses true-class probability (TCP). — [arXiv 1910.04851](https://arxiv.org/abs/1910.04851)
  - Self-Adaptive Training (SAT). — [arXiv 2002.10319](https://arxiv.org/abs/2002.10319)
- **Vazhentsev et al., ACL 2022.** ELECTRA/BERT on MRPC, CoLA and SST-2 (SST-2 subsampled to 10% to emulate low resource), plus CoNLL-2003 NER, scored by RCC-AUC.
  - Their best method was Mahalanobis distance with spectral normalization (MD-SN). It achieved "the best results on most of the datasets" and matched or beat deep ensembles and MC dropout. It "substantially outperforms all computationally expensive methods on the CoLA dataset, while on other text classification datasets it is on par with them".
  - SNGP gave substantial gains on MRPC and SST-2 but "significantly falls behind the trivial baseline" on NER.
  - MC dropout beats SR only when all dropout layers are active.
  - Caveat: the subsampled, low-resource setup inflates epistemic error, which favors density methods. — [ACL Anthology 2022.acl-long.566](https://aclanthology.org/2022.acl-long.566)
- **Vazhentsev et al., ACL 2023 (HUQ, ambiguous tasks).** ELECTRA/BERT on toxicity (ParaDetox, ToxiGen, Jigsaw), hate (Twitter, ImplicitHate), 20 Newsgroups, SST-5 and Amazon.
  - Existing UE methods mostly capture epistemic uncertainty and "often do no better than trivial baselines" on ambiguous tasks.
  - "The base epistemic UE methods sometimes cannot outperform even the weak SR baseline or even fall behind it by a large margin": RDE on Twitter and Amazon, DDU on Jigsaw and Amazon. "Vanilla DDU is significantly outperformed by SR for ELECTRA on Jigsaw."
  - HUQ combines ranks: it uses SR (aleatoric) inside the dense, ambiguous region and MD/DDU/RDE (epistemic) for low-density points, linearly mixing the ranks with α tuned on validation. HUQ-MD improved on both MD and SR on 6/8 datasets, with the largest gains on 20NG (−13.0% AUC-RC, ELECTRA) and ParaDetox (−4.9%). On ToxiGen it beat SR by 9.6% and MD by 10.6%. On some datasets HUQ did not beat SR or RDE, which the authors attribute to covariate shift.
  - Entropy behaved "very similar to SR". — [ACL Anthology 2023.acl-long.652](https://aclanthology.org/2023.acl-long.652) (PDF text read)
- **kNN-UE (arXiv 2407.02138).** Rescales the logits by W(ŷ) = α·mean_k exp(−d_k/τ) + λ(S(ŷ)/K + b), where S(ŷ) is the number of the K=32 nearest training neighbors that share the predicted label. Hyperparameters are fit by L-BFGS on validation loss. DeBERTaV3-base, E-AURC vs SR:

  | Dataset | Setting | SR | kNN-UE |
  |---|---|---|---|
  | IMDb | ID | 10.60 | 7.97 |
  | Yelp | OOD | 11.84 | 9.92 |
  | MNLI | ID | 31.29 | 23.83 |
  | SNLI | OOD | 37.62 | 34.97 |
  | OntoNotes bn | ID | 21.90 | 20.14 |

  - ECE dropped sharply (e.g. MNLI 8.36→1.41). AUROC changes were tiny (IMDb 98.35→98.64; SNLI 96.07→96.00, slightly worse).
  - The label-agreement term hurt E-AURC on NER (many labels).
  - Overhead was small for sentence-level tasks (MNLI 8.41s→9.08s total) and about 2× for token-level tasks.
  - Important: SR here was an uncalibrated baseline. Part of the E-AURC gain is the confidence rescaling, which a temperature-scaled baseline would partly share. Temperature scaling appeared as a separate baseline in the paper; I did not extract its E-AURC. — [arXiv 2407.02138](https://arxiv.org/html/2407.02138v2)
- **Post-hoc logit scores (Cattelan & Silva, UAI 2024).** Benchmarked more than 20 post-hoc confidence estimators on 84 ImageNet classifiers. MaxLogit-pNorm (p-norm-normalize the logits, then take the max logit) "completely fix[es] the pathological behavior" of "broken" MSP estimators. After that fix, selective performance is "almost entirely determined by" accuracy. It needs less than 1 holdout sample per class to tune, and the gains held under distribution shift. This is vision only; a well-calibrated small fine-tuned encoder may not have a "broken" MSP. — [arXiv 2305.15508](https://arxiv.org/abs/2305.15508v4)
- **Margin scores (Liang et al.).** Generalized selective classification under label and covariate shift. Arbitrary scaling of raw logits "can dramatically affect" SR-based selective performance. The authors propose raw-logit margin scores: the top-1 minus top-2 logit (RLconf-M) and a geometric margin. Margin scores reject samples near decision boundaries. — [arXiv 2405.05160](https://arxiv.org/pdf/2405.05160) (summary-level; venue not confirmed)
- **DOCTOR (Granese et al., NeurIPS 2021).** Black-box misclassification detector built from the softmax vector (e.g. 1 − Σp²). It claims to beat SOTA detectors on image and sentiment benchmarks, with up to 4% reduction in false-rejection rate in the partially-black-box mode, which uses input perturbation/gradients. — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html)
- **Large-scale vision study (Galil, Dabbah, El-Yaniv, ICLR 2023).** 523 ImageNet classifiers evaluated on AUROC, ECE, AURC and coverage at a selective-accuracy target. Distillation-based training "consistently" gives better uncertainty estimates. A ViT reached 99% top-1 selective accuracy at 47% coverage. — [ICLR 2023 poster](https://iclr.cc/virtual/2023/poster/11783); [arXiv 2206.02152](https://arxiv.org/abs/2206.02152v2)
- **Distance-aware single models.** [link-only]
  - SNGP: spectral normalization plus a GP output layer. — [arXiv 2006.10108](https://arxiv.org/abs/2006.10108)
  - DUQ — [arXiv 2003.02037](https://arxiv.org/abs/2003.02037)
  - DDU: GDA density for epistemic uncertainty, softmax entropy for aleatoric. — [arXiv 2102.11582](https://arxiv.org/abs/2102.11582)
  - Energy score, designed for OOD. — [arXiv 2010.03759](https://arxiv.org/abs/2010.03759)
  - Trust Score: ratio of the distance to the nearest other-class cluster to the distance to the predicted-class cluster. — [arXiv 1805.11783](https://arxiv.org/abs/1805.11783)
  - Deep ensembles — [arXiv 1612.01474](https://arxiv.org/abs/1612.01474)
  - MC dropout — [arXiv 1506.02142](https://arxiv.org/abs/1506.02142)
- **Sun et al. ICML 2022 (deep kNN).** Uses k-th-nearest-neighbor distance on normalized penultimate features for OOD detection, not misclassification detection of in-distribution inputs. [link-only] — [arXiv 2204.06507](https://arxiv.org/abs/2204.06507)

### Inferences
- For a well-fine-tuned, calibrated encoder on in-distribution data whose errors are mostly label flips, expect MSP, entropy, margin and DOCTOR to give nearly identical rankings. In binary tasks they are all monotone in |logit|, so they rank identically.
- Density and kNN signals are the only family that adds information MSP lacks. The literature says they help when errors come from sparse or shifted regions and can hurt in dense, ambiguous regions.
- Ensembles are the most robust improvement in vision benchmarks. They do not fit a 25 ms CPU budget beyond 2–3 tiny members, and MC dropout needs 20 or more passes to match SR (Xin et al.).
- Learned confidence heads are hype for our purpose. Feng et al. show the head's ranking is worse than the classifier's own softmax.

### Gaps
- I found no study that isolates "random test-label flips" as the failure source and compares scores. Theory (Q3) says no score can rank them.
- Jaeger et al.'s per-method table (DG, ConfidNet, DE, energy, Mahalanobis) was not extracted. Only the abstract-level conclusion is cited.
- Feng et al.'s numeric tables were not read.
- No source reports results for Ettin or ModernBERT-class 17M encoders specifically.

## 2. Metrics: AURC, E-AURC, risk@coverage, AUROC failure prediction, and how they relate to our metric

### Takeaway
Our metric is a single-point, normalized risk@coverage: one minus the relative risk reduction at 80% coverage. It depends only on how the confidence values rank cases within the 80% cut, and the base error rate matters a lot. E-AURC and AUGRC normalize for accuracy in different ways. AUROC of failure detection ignores accuracy. Global, monotone calibration does not change any ranking metric.

### Cited Findings
- **AURC and E-AURC.** AURC is the area under the selective-risk vs coverage curve. E-AURC = AURC − AURC*, where AURC* is the area for the optimal ranking given the same error rate. It was introduced to compare confidence functions across models with different accuracy. [link-only] — [Geifman et al., "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers", arXiv 1805.08206](https://arxiv.org/abs/1805.08206)
- **Jaeger et al. 2023** argue AURC is the metric that jointly reflects classifier performance and confidence ranking. — [arXiv 2211.15259](https://arxiv.org/abs/2211.15259)
- **Traub et al. (NeurIPS 2024)** set five requirements for multi-threshold selective classification metrics and show that existing metrics, including AURC, fail some of them. They propose AUGRC, the area under the generalized risk-coverage curve, interpretable as the average risk of undetected failures. AUGRC changed method rankings on 5 of 6 datasets in a benchmark of 13 confidence scoring functions. AURC can "erroneously favor" a method because of "excessive contributions of only few high-confidence failures". — [arXiv 2407.01032](https://arxiv.org/abs/2407.01032v2); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract.html)
- **Kamath et al. (2020)** used "coverage at a fixed accuracy", the % of questions answered while keeping 80% accuracy. Their calibrator answered 56% vs 48% for MaxProb. — [arXiv 2006.09462](https://arxiv.org/abs/2006.09462)
- **Our metric (task brief).** AbstScore = 1 − risk@0.8 / risk@1.0.

### Inferences
- Our metric is closely related to E-AURC: it is a normalized single slice of the curve. Its "high-confidence failure" sensitivity is the same pathology Traub et al. flag for AURC. A few confident errors (e.g. flipped labels on clear-cut cases) dominate it.
- **Upper bound.** Let e be the full-coverage error, of which a fraction η is irreducible noise that is independent of x and so spreads uniformly over confidence ranks. Even a perfect ranking of the remaining errors leaves about η in the retained 80%, so AbstScore ≤ 1 − η/e. Worse, if the flips fall on cases the model finds easy (high confidence in the original label), the noise concentrates in the retained set: risk@0.8 → η/0.8 and AbstScore ≤ 1 − 1.25η/e.
- Example: route e = 7% with η = 5.5% caps the route AbstScore near 0.21 (uniform) or even negative (concentrated). This is arithmetic, not a sourced figure; plug in your measured numbers.
- **Calibration and ranking.** Temperature scaling, σ(z/T), preserves the ordering of |z|, so max(p, 1−p) rankings for binary tasks are unchanged. A Platt bias term (σ(az+b), b≠0) can reorder cases across the two sides of 0.5. For multiclass, temperature scaling can reorder MSP slightly. Calibration work only moves the abstention metric through these side effects.
- **Base error matters.** If the judges' dataset has a different noise rate, AbstScore moves even with an identical model, because the denominator changes.

### Gaps
- I did not find a published name for "1 − risk@c / risk@1". It resembles the relative AURC improvement used informally in several papers.

## 3. Aleatoric vs epistemic uncertainty: when do kNN and density signals help, and when not?

### Takeaway
Selective prediction can only reject cases whose error is predictable from the input. MSP or entropy captures aleatoric (class-overlap) uncertainty that the model has learned. Density and kNN capture epistemic, low-density uncertainty. Neither can detect label noise that is independent of the input, such as random flips or labels driven by hidden account history. kNN agreement over training embeddings helps when (a) test inputs fall in sparse or shifted regions, or (b) training neighbors carry systematic label disagreement the classifier smoothed over. It cannot predict a test label flipped at random.

### Cited Findings
- **HUQ framing (ACL 2023).**
  - Uncertain inputs fall into two kinds of region. "a) OOD areas, which can be detected with epistemic UE methods" and "b) in-distribution ambiguous areas, for detection of which, aleatoric UE" is needed.
  - SR and entropy "mostly reflect the aleatoric one".
  - The proportion of ambiguous texts "can reach 90 percent" in some tasks (AIRI blog).
  - The best mixing weight on ToxiGen was α≈0.5. At α→0 (pure Mahalanobis) results were "even worse in terms of AUC-RC than the SR baseline". — [ACL 2023](https://aclanthology.org/2023.acl-long.652); [AIRI blog](https://airi.net/blog/HUQ/)
- **kNN-UE.** The gain comes from both distance and neighbor-label agreement. The label term hurts with many labels (NER). OOD gains are of similar size to ID gains (Yelp 11.84→9.92 E-AURC). — [arXiv 2407.02138](https://arxiv.org/html/2407.02138v2)
- **Vazhentsev 2022.** Mahalanobis+SN beats SR in a low-resource setting (SST-2 at 10% of the data), where epistemic error is large by construction. — [ACL 2022](https://aclanthology.org/2022.acl-long.566)
- **Label noise is common in real benchmarks.** Northcutt et al. (NeurIPS 2021 D&B) estimate at least 3.3% label errors on average across 10 test sets (CV, NLP, audio), and 6% on ImageNet val. About 51–54% of flagged candidates were human-confirmed (the figure differs across versions). Lower-capacity models can look better on noisy test sets. — [arXiv 2103.14749](https://arxiv.org/pdf/2103.14749)
- **kNN and training-set noise.** Deep k-NN filtering is used to detect noisy training labels (Bahri, Jiang & Gupta, ICML 2020). This is a training-time cleaning tool, not a test-time abstention signal. [link-only] — [arXiv 2004.12289](https://arxiv.org/abs/2004.12289)
- **DDU** explicitly separates feature-density (epistemic) from softmax entropy (aleatoric). [link-only] — [arXiv 2102.11582](https://arxiv.org/abs/2102.11582)

### Inferences
- **Route (random flips).** If the ~5–6% route errors are input-independent flips, no confidence score can rank them. kNN agreement on test inputs measures training-label consistency near x, not the test label's corruption. Expected gain is close to zero on our data.
  - Possible upside on judges' data: if their route errors are genuine boundary or OOD cases, kNN or Mahalanobis fused HUQ-style could cut AURC by a few to ~10% relative.
  - Risk: on dense, ambiguous data a pure-density score can make ranking worse. Only rank-mix with α tuned on validation, and keep SR as the backbone.
- **Approve (hidden account history).** Refusals caused by features the model cannot see are aleatoric from the model's view. kNN over text embeddings will not see them either.
  - Only new input features help, e.g. any account or metadata field available at inference.
  - So might a prior on base rates per segment, if the hidden refusal rate varies by observable segment such as amount bucket, category or customer tier.
  - Check whether error rate is predictable from such segments. If it is, add them as features or as a per-segment confidence shrink.
- **Cheap test of the teammate's idea.** Compute the AUROC of "kNN label disagreement" for predicting errors on a held-out split, conditioned on MSP decile. If AUROC within deciles is ≈0.5, kNN adds nothing beyond MSP.

### Gaps
- No source quantifies kNN/Trust-Score gains specifically under injected random label noise at test time for text. That is the case most relevant to us; theory and our arithmetic are the only basis.

## 4. Reject-option and learning-to-defer theory

### Takeaway
The Bayes-optimal reject rule is "abstain when the minimum conditional expected cost of acting exceeds the reject cost" (Chow 1970). Cost-based, bounded-coverage and bounded-risk formulations all share the same optimal strategy: rank by the conditional risk of the Bayes classifier and cut at the coverage or threshold you need (Franc, Prusa & Voracek, JMLR 2023). For a fixed 80% coverage target, the best possible score is a well-estimated posterior. Consistent surrogate losses (Mozannar & Sontag; Cortes, DeSalvo & Mohri; Mao, Mohri & Zhong) matter when training a rejector jointly or deferring to a human of known accuracy. They do not create information the posterior lacks.

### Cited Findings
- **Chow's rule.** Chow (1970), "On optimum recognition error and reject tradeoff": reject if max_y p(y|x) < 1 − c for 0/1 loss with reject cost c. [link-only] — [IEEE Trans. IT 16(1), DOI 10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406)
- **Franc, Prusa & Voracek (JMLR 24, 2023).**
  - The cost-based, bounded-improvement and bounded-abstention reject models "lead to the same prediction strategy": the Bayes classifier plus a randomized Bayes selection function.
  - The Bayes classifier abstains "when the conditional expected risk exceeds the reject cost".
  - They define a "proper uncertainty score" that is sufficient to build the selection function, with two Fisher-consistent algorithms to learn it for any black-box classifier.
  - The ICML 2019 precursor learns the uncertainty score discriminatively. — [JMLR](https://jmlr.org/papers/v24/21-0048.html); [arXiv 2101.12523](https://arxiv.org/abs/2101.12523); [ICML 2019](https://proceedings.mlr.press/v97/franc19a.html)
- **Mozannar & Sontag (ICML 2020).**
  - Learning to defer to an expert reduces to cost-sensitive learning with a (K+1)-class extended label space. Their surrogate is the first Bayes-consistent one and generalizes cross-entropy. Defer when the defer score is at least the maximum class score.
  - Later work showed it, and the one-vs-all surrogate (Verma & Nalisnick 2022), are not realizable H-consistent.
  - Mao et al. generalized it to arbitrary costs with H-consistency bounds. — [ICML 2020](https://icml.cc/virtual/2020/poster/53615); [arXiv 2006.01862](https://arxiv.org/abs/2006.01862); [Verma & Nalisnick, arXiv 2202.03673](https://arxiv.org/pdf/2202.03673); [Mao et al., realizable H-consistent L2D, arXiv 2407.13732](https://arxiv.org/pdf/2407.13732)
- **Score-based multi-class abstention.** Theory and consistent surrogates (Mao, Mohri & Zhong, 2023), extending Cortes, DeSalvo & Mohri's "learning with rejection" line. — [arXiv 2310.14770](https://arxiv.org/pdf/2310.14770)
- **Guarantees.** SGR (Geifman & El-Yaniv 2017) picks a threshold with a high-probability bound on selective risk. — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
- **Related calibration work.** [link-only]
  - Calibrated selective classification (Fisch et al., TMLR 2022) — [arXiv 2208.12084](https://arxiv.org/abs/2208.12084)
  - Learn-then-Test risk control (Angelopoulos et al.) — [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)

### Inferences
- Because our scorer ranks by max(p, 1−p) of the reported probability, the "proper uncertainty score" is whatever p we emit. Any extra signal, such as kNN or a segment prior, must be folded into p by shrinking it toward 0.5 or toward the class prior. kNN-UE's logit rescaling W(ŷ) is exactly such a mechanism; it keeps the argmax.
  - Shrinking p also changes the approve decision at the 0.75 threshold. Folding a signal into p is only coherent if it reflects a genuinely better posterior estimate.
- Chow-style per-decision thresholds tuned at a coverage floor of 0.85 already implement the optimal rule for a fixed scorer. Gains must come from a better p, not a better threshold rule.

### Gaps
- I did not verify Cortes, DeSalvo & Mohri (ALT 2016) directly. It is cited through Mao et al.

## 5. Cost-sensitive binary decisions: Bayes thresholds, cost curves, decision curves, industry practice and quant analogies

### Takeaway
With calibrated p = P(approve is correct), cost 3 for a wrong approval and cost 1 for a wrong hold, Bayes-optimal action is: approve iff 3(1−p) < p, i.e. p > 0.75 (Elkan 2001). Our threshold is already optimal if p is calibrated. Adding a reject option with cost d gives three zones: approve if 3(1−p) ≤ min(p, d); hold if p ≤ min(3(1−p), d); otherwise send to human review. Industry fraud and credit systems use the same two-threshold structure (auto-approve / manual review / auto-decline) and tune it against historical outcomes.

### Cited Findings
- **Elkan (IJCAI 2001).**
  - For two classes, the optimal decision uses the threshold p* = C_FP / (C_FP + C_FN). The optimal threshold is constant across subgroups.
  - Changing the class balance of training data has little effect on Bayesian or tree learners, so it is better to apply the cost matrix to probability estimates directly. Equivalently, the data can be rebalanced so a 0.5 threshold mimics p*.
  - Sources differ on the C(i,j) index convention. — [mlanthology](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations); restated in [arXiv 2406.09548](https://arxiv.org/pdf/2406.09548)
- **Stripe Radar.** Scores charges 0–99, with a default manual-review threshold of 65 and block threshold of 75. Moving the block threshold moves the review threshold. Merchants can backtest a threshold against past payments. Stripe's explainer says thresholds are tuned on fraud rate, false declines and customer impact, with step-up verification as a middle option. — [Stripe docs](https://docs.stripe.com/radar/risk-settings.md); [Stripe explainer](https://stripe.com/fr-ca/resources/more/fraud-scores-explained); third-party commentary that 65/75 are Stripe defaults, not an industry standard: [chargeback.io](https://www.chargeback.io/blog/fraud-score-glossary)
- **Decision curve analysis (Vickers & Elkin, Med Decis Making 2006).** Net benefit = TP/N − FP/N · p_t/(1−p_t). The threshold probability p_t encodes the harm:benefit ratio; p_t = 0.75 corresponds to 3:1. [link-only] — [DOI 10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361)
- **Selective QA abstention as "answer only if confident enough"** for user-facing systems, targeting 80% accuracy. — [Kamath et al. 2020](https://arxiv.org/abs/2006.09462)

### Inferences
- **Our approve rule.** The Chow-with-costs rejection score for approve is r(p) = min(3(1−p), p), the expected cost of the best action. It peaks at p = 0.75 (r = 0.75) and falls to 0 at p = 0 and p = 1.
  - The scorer ranks by max(p, 1−p), which is lowest at p = 0.5, not 0.75. Cases with p ∈ (0.5, 0.75) are held but look "confident" to the scorer.
  - If the scorer's error for approve is 0/1 decision mismatch, those cases are wrong with probability p > 0.5 yet ranked as mid-confidence. Abstaining on them first, as r(p) prescribes, is right for both cost and risk@coverage.
  - Our rule "min(3(1−p), p) − λp" tilts abstention further toward high-p cases. Check that λ is tuned on the same scorer.
- **Mismatch with the scorer.** If the official scorer counts errors as 0/1 against the gold label with ranking by max(p, 1−p), the risk-optimal abstention for approve is to abstain on the lowest P(decision correct). That is p for approve decisions and 1−p for holds, which is not the same as max(p, 1−p). Abstaining by our own P(decision correct) rather than by max(p, 1−p) is therefore consistent with risk minimization. The ranking the scorer applies still determines which of our abstentions count, so verify against the offline scorer.
- **Quant analogy.** Expected-value maximization under asymmetric loss is the same calculation as Elkan's threshold. Kelly-style sizing applies to repeated bets with variable stake and has no direct analogue for a fixed-cost binary approve/hold. The useful idea is "only act when the edge, EV = p·gain − (1−p)·loss, exceeds a margin covering model error". That supports a small safety margin above 0.75 when p is miscalibrated under shift. This is unsourced reasoning.
- **Human-review queue.** Industry uses a band (e.g. Stripe's 65–75) between auto-accept and auto-decline. That matches our three-zone rule, with the review-queue size fixed by the coverage floor.

### Gaps
- I found no public, quantitative writeup from a credit or insurance lender on how auto-approval thresholds and review-queue sizes are chosen, beyond Stripe's defaults.
- I did not fetch a Kelly criterion primary source.

## 6. Abstention under distribution shift, and calibrators trained to predict errors

### Takeaway
MSP-based abstention degrades under shift because fine-tuned models are overconfident on OOD inputs. Kamath et al. show that a small calibrator, a random forest over model-derived features trained to predict errors, beats MaxProb under domain shift (56% vs 48% coverage at 80% accuracy). Later work finds the gains are task-dependent, and MaxProb stays hard to beat across settings. Post-hoc logit normalization (pNorm) and margin scores are cheap, shift-robust alternatives in vision.

### Cited Findings
- **Kamath, Jia & Liang (ACL 2020).** On mixtures of SQuAD and five other QA datasets, the calibrator "answers 56% of questions while maintaining 80% accuracy" vs 48% for MaxProb. Softmax abstention fails because "the QA model is overconfident on out-of-domain inputs". The calibrator benefits from seeing OOD data, even from a domain different from the test domain. — [arXiv 2006.09462](https://arxiv.org/abs/2006.09462)
- **Varshney et al. (2022).** Across IID, OOD and adversarial settings, no method consistently beats MaxProb. Their follow-up calibrator improves on MaxProb for NLI and duplicate detection in IID and OOD settings. — [arXiv 2203.00211](https://arxiv.org/pdf/2203.00211); [RepL4NLP 2022](https://preview.aclanthology.org/setup/2022.repl4nlp-1.23)
- **Cattelan & Silva.** MaxLogit-pNorm gains are "preserved even under distribution shift" (ImageNet variants). — [arXiv 2305.15508](https://arxiv.org/abs/2305.15508v4)
- **Generalized selective classification (Liang et al.).** Under label and covariate shift, margin-based raw-logit scores beat SR-based ones because SR is sensitive to logit scale. — [arXiv 2405.05160](https://arxiv.org/pdf/2405.05160) (summary-level)
- **HUQ.** Gains shrink or vanish where there is "large covariate shift between the training and test data". — [ACL 2023](https://aclanthology.org/2023.acl-long.652)
- **kNN-UE.** Kept part of its E-AURC advantage OOD (SNLI 37.62→34.97; OntoNotes nw 81.49→74.27). — [arXiv 2407.02138](https://arxiv.org/html/2407.02138v2)
- **Jaeger et al.** Evaluated across shift types, SR was still the best overall. — [arXiv 2211.15259](https://arxiv.org/abs/2211.15259)

### Inferences
- If judges use a different dataset, thresholds tuned to a fixed confidence value will give a different coverage. Our coverage floor of 0.85 could be breached in either direction.
  - Safer: set thresholds by quantile at inference time so coverage stays at about 0.80–0.85 on the scored batch, if batch-level processing is allowed. Or tune thresholds on a mixture of in-domain and perturbed or paraphrased data, following Kamath's lesson of training on some shift.
- A Kamath-style error calibrator, a tiny GBM or logistic model over [MSP, margin, entropy, kNN distance, kNN agreement, length, segment features] trained to predict "decision wrong", is the principled way to fuse the teammate's kNN signal. It falls back toward MSP when the extra features carry no information.
  - Its output must be mapped into the emitted p to affect a max(p, 1−p) ranking.
  - Train it on held-out folds, not on training data, since kNN on training points is leaky.

### Gaps
- I found no NLP-specific study of how selective classification degrades under shift for tiny (under 50M) encoders.

## 7. Assessment for Decide: what could raise risk@80% vs what is hype

### Takeaway
Expect at most modest abstention gains from new confidence scores. The ceiling is set by input-independent noise (AbstScore ≤ 1 − η/e). Highest expected value:
1. Make abstention follow P(decision correct), using the cost-aware three-zone rule. For approve, this targets the p ∈ (0.5, 0.75) band.
2. Add any observable feature that predicts hidden-history refusals.
3. Fuse kNN or density only through a held-out-trained error calibrator or HUQ-style rank mix, accepting only if it beats MSP on held-out risk@80%.
4. Use quantile-based thresholds so coverage stays near 0.8–0.85 on judges' data.

Learned selection heads, MC dropout, energy and standalone Mahalanobis or kNN are hype for this setting.

### Cited Findings
- SR/MSP is the overall best performer in the largest failure-detection benchmark. — [Jaeger et al. 2023](https://arxiv.org/abs/2211.15259)
- No NLP method consistently beats MaxProb. — [Varshney et al. 2022](https://arxiv.org/pdf/2203.00211)
- Selective heads lose to the classifier's own softmax. — [Feng et al. 2023](https://arxiv.org/abs/2206.09034)
- Epistemic-only scores can fall well behind SR on ambiguous text, while hybrid rank-mixing gives 2–13% relative AUC-RC gains. — [HUQ, ACL 2023](https://aclanthology.org/2023.acl-long.652)
- kNN-UE improves E-AURC by roughly 8–25% relative over uncalibrated SR on DeBERTa sentence tasks, with small overhead. — [arXiv 2407.02138](https://arxiv.org/html/2407.02138v2)
- MC dropout needs at least 20 passes to match SR in NLP, so it is infeasible in 25 ms. — [Xin et al. 2021](https://aclanthology.org/2021.acl-long.84.pdf)
- Error-ranking regularization during fine-tuning improves selective AUC at no inference cost. — [Xin et al. 2021](https://aclanthology.org/2021.acl-long.84.pdf)
- Entropy-minimization regularization combined with SR gives SOTA selective classification in vision. — [Feng et al. 2023](https://arxiv.org/abs/2206.09034)
- A learned error calibrator beats MaxProb under shift (56% vs 48% coverage at 80% accuracy). — [Kamath et al. 2020](https://arxiv.org/abs/2006.09462)

### Inferences
Ranked candidates, cost and expected effect:
1. **Low cost, likely positive.** Abstain by P(decision correct) with the cost-aware score r(p) = min(3(1−p), p) rather than by max(p, 1−p), and verify against the offline scorer. The p ∈ (0.5, 0.75) hold band is where 0/1 errors concentrate.
2. **Low cost, maybe positive.** Replace temperature-scaled MSP for route with margin (top-1 minus top-2) or MaxLogit-pNorm, choosing on held-out risk@80%. In binary tasks margin equals MSP, so there is no change for approve or grounded.
3. **Medium cost, data-dependent.** A Kamath-style error calibrator over [MSP, margin, kNN distance, kNN label agreement (K≈16–32), text length, observable approve segment features], trained on out-of-fold predictions and folded back into p. Accept only if held-out risk@80% improves by more than its bootstrap CI. On our noise-dominated data expect roughly 0 to 0.03 AbstScore. On judges' data with genuine OOD cases there may be more upside.
4. **Training-time, medium cost.** Xin et al.'s error-ranking regularizer or Feng et al.'s entropy regularizer during fine-tuning. Both have zero inference cost.
5. **Hype for us.** SelectiveNet, Deep Gamblers, SAT and ConfidNet heads; MC dropout; energy score (an OOD tool); standalone Mahalanobis, kNN or Trust Score as the sole confidence score; large ensembles (over the latency budget). Also "better calibration" via plain temperature scaling, which leaves binary rankings unchanged.
6. **Measure the ceiling first.** Estimate η, the error that is unpredictable from the input, by re-labelling a sample or checking duplicates and near-duplicates with conflicting labels. Then compute 1 − η/e per decision. If route's ceiling is about 0.2–0.3, spend effort on approve and grounded, or on coverage and quality, rather than route abstention.

### Gaps
- The judges' scoring details for approve (0/1 vs cost-weighted error; whether confidence uses the emitted p) are unknown. They determine whether item 1 helps or is neutral.
- No primary source measures these techniques on Ettin-17M.
