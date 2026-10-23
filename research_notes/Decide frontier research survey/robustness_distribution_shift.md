# Robustness to distribution shift for a small fine-tuned text decision model (Decide: route / approve / grounded)

Scope note: researched 2026-10-24. Sources marked **[fetched]** were opened and read in this session (abstract or full text). Sources marked **[cited from prior knowledge, not re-fetched]** are well-known primary papers whose claims I state only at abstract level; the report writer should treat their wording as lower-confidence. Inference bullets are my reasoning applied to Decide (Ettin-17M, 6,500 template tickets, dev 85.2, CPU p95 < 25 ms).

---

## 1. How fine-tuned encoders degrade under domain/template shift, and which training-time interventions help

### Takeaway
Pretrained transformers lose less accuracy under shift than older models (a few points rather than 5–26 points), but they still lose accuracy, and distilled/small models lose more. The interventions with the best evidence per unit of effort are: (a) more diverse training text (LLM paraphrases or personas, public support corpora), (b) keeping the pretrained features intact (LP-FT, WiSE-FT weight interpolation, lower learning rate, fewer epochs), and (c) contrast/counterfactual examples that break template shortcuts. LLM-generated data helps most when the classes are clearly distinct and least when they are near-synonyms or subjective.

### Cited Findings
**Degradation evidence**
- Hendrycks et al. 2020 (ACL), "Pretrained Transformers Improve OOD Robustness": the OOD performance drops of pretrained transformers are "substantially smaller" than those of BoW, ConvNets and LSTMs. Larger models are *not* necessarily more robust, distillation can hurt robustness, and more diverse pretraining data helps. — [arXiv 2004.06100](https://arxiv.org/abs/2004.06100) [fetched]
- Table 4 of the same paper (train on SST-2 or IMDb, test on the other): BERT-base 91.9 → 87.5 (−4.4) and 91.8 → 87.6 (−4.3). BERT-large 93.6 → 88.3 (−5.3). RoBERTa 95.6 → 92.8 (−2.8) and 94.3 → 91.0 (−3.4). For comparison, BoW 80.6 → 73.9 (−6.8) and 85.9 → 78.3 (−7.6), and LSTM-word2vec 89.9 → 63.7 (−26.3). Figure 2 and Section 4 say "model distillation can reduce robustness, as evident in our DistilBERT results." Figure 3 says the IID/OOD gap "is not improved with larger models, unlike in computer vision." — [arXiv 2004.06100 PDF](https://arxiv.org/pdf/2004.06100) [fetched, numbers read from extracted tables]
- Contrast sets (Gardner et al. 2020): small, label-changing perturbations by experts were written for 10 NLP datasets. Model performance on them is markedly lower than on the original test sets, "up to 25% in some cases". This shows that i.i.d. test accuracy overstates what models have learned. — [arXiv 2004.02709](https://arxiv.org/abs/2004.02709) [fetched]
- CheckList (Ribeiro et al. 2020): held-out accuracy "often overestimates" NLP model performance. In a user study, practitioners with CheckList wrote about 2× as many tests and found almost 3× as many bugs. A commercial sentiment team found new actionable bugs in a heavily tested model. — [arXiv 2005.04118](https://arxiv.org/abs/2005.04118) [fetched]

**Keeping pretrained features (LP-FT, WiSE-FT, soups)**
- Kumar et al. 2022 (ICLR), LP-FT: across 10 distribution-shift datasets, full fine-tuning gave on average about 2% higher ID accuracy but about 7% *lower* OOD accuracy than linear probing. LP-FT (train the head first with the backbone frozen, then fine-tune everything) gives about 1% better ID and about 10% better OOD than full fine-tuning. The mechanism: a randomly initialised head sends large gradients that distort the lower-layer features. — [arXiv 2202.10054](https://arxiv.org/abs/2202.10054) [fetched]
- Wortsman et al. 2022 (CVPR), WiSE-FT: linearly interpolating the zero-shot (pretrained) weights with the fine-tuned weights improves accuracy under shift by 4–6 pp on ImageNet shifts, and by 2–23 pp on six further shifts. ID accuracy improves by 0.8–3.3 pp on seven transfer datasets, "at no additional computational cost during fine-tuning or inference." — [arXiv 2109.01903](https://arxiv.org/abs/2109.01903) [fetched]
- Caveat: the WiSE-FT and LP-FT headline numbers are mostly from vision (CLIP and ImageNet-type shifts). WiSE-FT needs a meaningful "zero-shot" endpoint. For an encoder like Ettin with a new classification head there is no zero-shot head, so the natural analogue is interpolating between the LP (linear-probe) checkpoint and the FT checkpoint. — inference from [arXiv 2109.01903](https://arxiv.org/abs/2109.01903) and [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)
- Model soups (Wortsman et al. 2022, ICML): averaging the weights of several fine-tunes with different hyperparameters improves accuracy and robustness with no extra inference cost. — [arXiv 2203.05482](https://arxiv.org/abs/2203.05482) [cited from prior knowledge, not re-fetched]

**Domain-adaptive pretraining**
- Gururangan et al. 2020 (ACL), "Don't Stop Pretraining": continued pretraining on in-domain unlabeled text (DAPT) improves performance in high- and low-resource settings across 4 domains and 8 classification tasks. Task-adaptive pretraining on the task's own unlabeled text (TAPT) also helps, and its gains persist after DAPT. Simple data selection to enlarge the task corpus is an effective alternative when DAPT is too expensive. — [arXiv 2004.10964](https://arxiv.org/abs/2004.10964) [fetched]

**Data augmentation and synthetic data**
- EDA (Wei & Zou 2019) uses four operations: synonym replacement, random insertion, random swap and random deletion. Gains were largest on small datasets. On average, EDA with 50% of the training data matched normal training on 100%. This was shown for CNNs and RNNs, not for pretrained transformers. — [arXiv 1901.11196](https://arxiv.org/abs/1901.11196) [fetched]
- Sahu et al. 2022, data augmentation for intent classification with off-the-shelf LLMs (GPT-3), four intent datasets, few-shot. Generated data helps noticeably when intents are clearly distinct. It is much less useful for semantically close intents, because the LLM produces utterances that actually belong to a neighbouring intent. Preliminary evidence suggests an LLM-as-classifier relabel/filter step improves quality. — [arXiv 2204.01959](https://arxiv.org/abs/2204.01959) [fetched]
- Li et al. 2023 (EMNLP), "Synthetic Data Generation with LLMs for Text Classification: Potential and Limitations": LLM-synthetic data is inconsistently effective across tasks. Task-level and instance-level *subjectivity* are negatively associated with the performance of models trained on synthetic data. — [arXiv 2310.07849](https://arxiv.org/abs/2310.07849) [fetched]
- ZeroGen (Ye et al. 2022): generate a labelled dataset from scratch with a PLM, then train a tiny task model (e.g. an LSTM) with "orders of magnitude fewer parameters" than the generator. Evaluated on text classification, QA and NLI. — [arXiv 2202.07922](https://arxiv.org/abs/2202.07922) [fetched; abstract gives no numbers]
- AugGPT (Dai et al. 2023): ChatGPT rephrases each training sentence into several variants for few-shot text classification. Reported to beat EDA and back-translation baselines. — [arXiv 2302.13007](https://arxiv.org/abs/2302.13007) [cited from prior knowledge, not re-fetched]
- Persona-driven generation (Persona Hub, Ge et al. 2024): conditioning an LLM on 1 billion web-derived personas to diversify synthetic data. — [arXiv 2406.20094](https://arxiv.org/abs/2406.20094) [cited from prior knowledge, not re-fetched]
- Counterfactually augmented data (Kaushik et al. 2020, ICLR): humans minimally edit examples to flip the label, which reduces reliance on spurious features. — [arXiv 1909.12434](https://arxiv.org/abs/1909.12434) [cited from prior knowledge, not re-fetched]

**Public support or intent data (licences)**
- Bitext customer-support dataset: **CDLA-Sharing-1.0**, 26,872 rows. The card says 27 intents in 10 categories (ACCOUNT, CANCELLATION_FEE, DELIVERY, FEEDBACK, INVOICE, NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING_ADDRESS). It carries 12 language-variation tags: morphological and semantic variation, interrogative, coordinated, negation, politeness, colloquial, offensive, keyword mode, abbreviations, errors/typos. It is *synthetic*: seeds extracted from natural text, expanded by NLG, curated by linguists. — [HF: bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) [fetched]
- BANKING77: **CC-BY-4.0**, 13,083 queries, 77 fine-grained intents (train 10,003 / test 3,080), single domain. Several intents are near-duplicates (e.g. getting_spare_card vs getting_virtual_card). The card recommends pairing it with coarser multi-domain sets. — [HF: PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) [fetched]; source paper Casanueva et al. 2020 — [arXiv 2003.04807](https://arxiv.org/abs/2003.04807) [fetched]
- CLINC150 (Larson et al. 2019): 150 intents across 10 domains plus out-of-scope queries. Useful as negative/OOS data for an abstention gate. — [arXiv 1909.02027](https://arxiv.org/abs/1909.02027) [cited from prior knowledge, not re-fetched; licence to verify]
- ABCD (Chen et al. 2021, NAACL): over 10K human-to-human customer-service dialogues with policy-dependent actions. Relevant to "approve" because agents follow company policies. — [arXiv 2104.00783](https://arxiv.org/abs/2104.00783) [cited from prior knowledge, not re-fetched; licence to verify]

### Inferences
- Ettin-17M is far smaller than BERT-base. The finding that DistilBERT is less robust suggests that Decide's tiny encoder sits at the fragile end, so a drop larger than 3–5 pts under real shift is plausible. Template-generated training data makes this worse, because the model can memorise template tokens.
- The cheapest high-value change is a **diversity injection**. Paraphrase each training ticket with a permissively licensed local LLM, offline at training time, into several registers: typos, colloquial, terse "keyword mode", long rambling, negation, politeness. These are the same axes Bitext tags. Then relabel/filter with a stronger classifier (Sahu et al.). For **route**, mapping Bitext's 27 intents onto Decide's 5 queues gives real lexical variety, with a licence caveat: CDLA-Sharing allows use, and its share-alike obligations apply to the *data*, not to model outputs. Have someone check before redistributing.
- For "approve" and "grounded", the label depends on numbers and on KB content, not on style. Paraphrasing must keep the amount, tier and evidence unchanged. Generating contrast pairs (same ticket, amount moved across the cap, opposite label) is the analogue of contrast sets and counterfactual augmentation, and should stop the model keying on template phrasing.
- Training recipe changes that cost nothing at inference: LP-FT (freeze the encoder and fit the heads, then unfreeze with a low LR), a weight average of LP and FT checkpoints (WiSE-style α sweep on a *shifted* validation split), and a soup of 3–5 seeds. Each keeps the 25 ms budget because inference is still a single model.
- TAPT/DAPT: a few thousand MLM steps on unlabeled support text (Bitext + BANKING77 + CLINC text) before fine-tuning is cheap at 17M parameters and directly follows Gururangan et al.

### Gaps
- I found no study that measures LP-FT or WiSE-FT specifically on sub-50M-parameter text encoders, or on template-to-real ticket shift.
- No numeric results were obtained for ZeroGen, AugGPT or Sahu et al. beyond their abstracts.
- Licences for CLINC150, ABCD and the Kaggle "Customer Support on Twitter" corpus were not verified in this session. I recall the Twitter corpus as non-commercial (CC BY-NC-SA); verify before using it.
- MultiWOZ is task-oriented booking dialogue, not support routing. Its relevance is low and it was not researched.

---

## 2. Zero-/few-shot fallbacks without retraining, and gating them by OOD detection

### Takeaway
NLI-based and GLiClass zero-shot classifiers work for coarse, clearly named labels but collapse on fine-grained intents: GLiClass-edge scores F1 0.027 on BANKING77. They are also 2–25× bigger than Ettin-17M, so they will likely break the 25 ms CPU budget unless gated to a small fraction of traffic. The fine-tuned model's own max-softmax probability is already a reasonable OOD signal: pretrained transformers reach AUROC of about 88 versus about 50–65 for older models. So use it as the gate, and make the fallback *abstain/escalate* rather than replace the fine-tuned prediction.

### Cited Findings
- Hendrycks et al. 2020, OOD detection with the maximum softmax probability (MSP): mean AUROC is 88.4 (BERT-large) and 88.7 (RoBERTa), versus 54.2 (BoW), 51.8 (avg-word2vec) and 49.3 (LSTM-word2vec). Mean FAR95 is 30.5 (BERT-large) and 43.0 (RoBERTa) versus 88.4 (BoW). Older models are "frequently more confident when classifying OOD examples than when classifying IID test examples." — [arXiv 2004.06100 PDF, Tables 7–8](https://arxiv.org/pdf/2004.06100) [fetched]
- MoritzLaurer/deberta-v3-base-zeroshot-v2.0: MIT licence, about 0.2B parameters, max 512 tokens. Mean macro-F1 on 28 classification datasets is about 0.643 zero-shot, rising to about 0.834 with few-shot training. For comparison, bart-large-mnli scores about 0.497 and the commercially clean "-c" variant about 0.619 (table alignment read with some uncertainty). Models without "-c" were trained on data that includes non-commercial licences (ANLI, WANLI, LingNLI). The "-c" variants use Mixtral-synthetic data plus MNLI and FEVER-NLI. — [HF model card](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) [fetched]
- GLiClass v3 (knowledgator), Apache-2.0. gliclass-edge-v3.0 has 32.7M parameters, average zero-shot F1 about 0.49, and per-dataset F1 of SST-2 0.82, IMDb 0.85, AG News 0.66, SNIPS 0.53, **BANKING77 0.027**. It classifies all labels in a single forward pass. Throughput is about 97 ex/s at batch size 1 on an A6000 GPU, falling to about 83 ex/s with 128 labels. Larger versions: base-v3.0 (187M, F1 0.656), large-v3.0 (439M, F1 0.700). — [HF: knowledgator/gliclass-edge-v3.0](https://huggingface.co/knowledgator/gliclass-edge-v3.0) [fetched]
- SetFit (Tunstall et al. 2022): contrastive fine-tuning of a Sentence Transformer on a few text pairs, then a classification head. Comparable to PEFT/PET with "orders of magnitude less parameters" and "an order of magnitude faster to train." — [arXiv 2209.11055](https://arxiv.org/abs/2209.11055) [fetched]
- BANKING77 paper: dual-sentence-encoder intent detectors (USE, ConveRT) beat fine-tuned BERT-large, with gains "especially pronounced" at 10 or 30 examples per intent. — [arXiv 2003.04807](https://arxiv.org/abs/2003.04807) [fetched]

### Inferences
- For **route** (5 coarse queues), a label-description fallback is cheap and can be distilled into the main model. Embed queue descriptions plus a few exemplars with Ettin's own pooled embeddings (nearest-centroid, as in SetFit/T3A). This costs nearly nothing at inference because the embedding is already computed. A separate 0.2B DeBERTa NLI model needs one forward pass per label (5 passes), which is very likely over 25 ms p95 on CPU. Not measured; benchmark before adopting.
- For **approve**, zero-shot NLI is the wrong tool. Approval depends on numeric caps and policy, which NLI models were not trained to compare (see Section 4).
- For **grounded**, an NLI model *is* the natural zero-shot fallback (premise = KB article, hypothesis = draft claim). It is the most defensible place to spend extra latency, but only on low-confidence cases.
- A practical gate: combine the fine-tuned MSP (or energy), a Mahalanobis or kNN distance in Ettin embedding space to the training set, and input-shape checks (length, unparsed amount, unknown tier). Then calibrate (temperature scaling) on a *shifted* validation split. If the gate fires, either abstain or blend with the prototype/fallback score. Hendrycks's AUROC of about 88 for MSP comes from base/large models; expect lower for a 17M model, and measure it on held-out templates (Section 5).
- Distil, don't deploy: run the large zero-shot NLI model or a local LLM *offline* to pseudo-label diverse synthetic and public text, then train Ettin on it. The robustness transfers to the student without the latency cost. This is ZeroGen-style.

### Gaps
- I found no CPU latency benchmarks for the MoritzLaurer or GLiClass models. GLiClass throughput is GPU-only.
- I found no paper that evaluates "MSP-gated blending of fine-tuned and zero-shot heads" for intent classification with numbers.

---

## 3. Test-time adaptation (TENT, T3A, TTT) and label-shift adaptation (EM / BBSE)

### Takeaway
Entropy-minimisation TTA (TENT) is unreliable for text. In the one NLP benchmark found, it gave +0.5 to +1.8 EM on XLM-R-base but −4 to −12.6 EM on XLM-R-large. It is hyperparameter-sensitive, and entropy minimisation sharpens predictions, which tends to hurt calibration. Skip it for a calibrated, abstaining system. Label-shift correction with EM prior re-estimation (Saerens 2002) is cheap, needs only unlabeled test predictions, and works well *if the model is calibrated first* (Alexandari et al. 2020). But it only fixes changed class frequencies, not new phrasings or new amount caps.

### Cited Findings
- Ye et al. 2022 (Findings of EMNLP), ColdQA benchmark (text corruption, language change, domain change). EM gain of Tent over the source model: XLM-R-base +1.1 / +1.4 / +1.7 / +1.8 / +0.5. XLM-R-large **−4.0 / −12.6 / −9.9 / +0.1 / −1.2** (NoiseQA-na 65.55 → 52.91). Pseudo-labelling (PL) and their OIL method were positive (e.g. OIL +3.5 to +6.0 on XLM-R-large). Tent is "much less effective than OIL and PL on ColdQA." PL needed a 1e-7 learning rate on some subsets and showed negative gains late in continual adaptation. The paper does not discuss calibration. — [ar5iv 2302.04618](https://ar5iv.labs.arxiv.org/html/2302.04618); [ACL Anthology 2022.findings-emnlp.460](https://preview.aclanthology.org/new-sigs/2022.findings-emnlp.460/) [fetched]
- Entropy-minimisation TTA and calibration: Bar, Shaer & Romano (NeurIPS 2024) depart from entropy minimisation with online *entropy matching* (a betting martingale that adapts only when shift is detected). They report better test-time accuracy under shift while keeping accuracy and calibration when there is no shift. — [arXiv 2408.07511](https://arxiv.org/abs/2408.07511); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract.html) [search-result level]
- Work on test-time prompt tuning argues that standard entropy minimisation "indiscriminately sharpens predictions," raising confidence even on wrong predictions. — [arXiv 2608.30230](https://arxiv.org/pdf/2608.30230) [search snippet only; arXiv ID not independently verified, treat as weak]
- TENT (Wang et al. 2021, ICLR): updates only normalisation affine parameters to minimise prediction entropy on test batches. — [arXiv 2006.10726](https://arxiv.org/abs/2006.10726) [cited from prior knowledge, not re-fetched]. Note: TENT was designed around BatchNorm statistics. Transformers use LayerNorm with no batch statistics, and Decide serves single requests, so per-request batch adaptation is ill-posed.
- Alexandari, Kundaje & Shrikumar 2020 (ICML), "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation". The Saerens et al. EM/ML method corrects label shift without retraining but assumes p(y|x) is calibrated, "which is not true of modern neural networks." BBSL/BBSE and RLLS handle miscalibration but need retraining with importance weights. ML plus bias-corrected calibration (temperature scaling with per-class bias) outperforms BBSL and RLLS across datasets and shifts. The objective is concave. — [arXiv 1901.06852](https://arxiv.org/abs/1901.06852) [fetched]
- Saerens, Latinne & Decaestecker 2002 (Neural Computation), EM re-estimation of class priors on unlabeled test data. — [doi 10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446) [cited from prior knowledge, not re-fetched]
- Lipton, Wang & Smola 2018 (ICML), BBSE: estimates the test label distribution from the confusion matrix of a black-box predictor. Consistent under the label-shift assumption. — [arXiv 1802.03916](https://arxiv.org/abs/1802.03916) [cited from prior knowledge, not re-fetched]

### Inferences
- Label-shift EM is the one adaptation that fits Decide's constraints. It is unlabeled, closed-form-ish, adds no inference latency (re-weight logits by π_test/π_train), and preserves calibration under the label-shift assumption once temperature/bias-calibrated. Use it if the judges' set arrives in bulk or the API sees a stream. Example: a new dataset with a different queue mix, or a different approve:deny ratio.
- For **approve**, label-shift correction changes the decision threshold implied by the 3:1 cost. After re-estimating priors, recompute the Bayes threshold (approve if p(approve) > 3/(3+1) = 0.75, adjusted for the priors) instead of keeping a fixed tuned threshold.
- Label-shift correction does *not* fix concept shift. If the hidden set uses different amount caps, p(y|x) itself changes, and no unlabeled method can recover the new caps without at least a few labels or an explicit policy input.
- T3A-style prototype adjustment (classifier-head-only, no backprop) is closer to safe than TENT, but I found no NLP evidence for it in this session.

### Gaps
- I found no paper measuring calibration (ECE) after TENT on text classifiers specifically. The calibration claims above come from vision/VLM work and the NeurIPS 2024 entropy-matching paper.
- I did not retrieve T3A (Iwasawa & Matsuo, NeurIPS 2021) numbers for NLP.

---

## 4. Robust feature design for numbers and amount rules

### Takeaway
Hard-coded normalisers and fixed tier × amount buckets encode the training generator's conventions ("a month = 30 days", specific caps). They will silently fail on new formats or new caps. Make the *parser* broad and tested, and make the *rule* learnable and monotone: approval probability must not increase with amount within a tier. Expose the cap as a learned, per-tier threshold rather than an embedded token pattern.

### Cited Findings
- Gradient-boosted trees support per-feature monotone constraints: LightGBM `monotone_constraints` — [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html); scikit-learn `HistGradientBoostingClassifier(monotonic_cst=...)` — [scikit-learn docs](https://scikit-learn.org/stable/modules/ensemble.html#monotonic-constraints) [cited from prior knowledge, not re-fetched]
- Contrast sets and counterfactual edits (Section 1) show that models exploit superficial cues unless training includes minimally different examples with flipped labels. — [arXiv 2004.02709](https://arxiv.org/abs/2004.02709) [fetched]; [arXiv 1909.12434](https://arxiv.org/abs/1909.12434) [not re-fetched]

### Inferences
- Split **approve** into (i) an extractor (amount, currency, tier, period) that is deterministic and heavily tested on varied formats, and (ii) a small monotone model over [log(amount), tier one-hot, encoder-derived risk features]. Examples of formats: "$1,234.50", "1.2k", "USD 300", "300 bucks", "three hundred", "€", "per month", ranges, multiple amounts. The model should output a calibrated p(approve). Unlike fixed buckets, a monotone GBM or a logistic model with tier-specific intercepts on log(amount) learns the cap location from data. If judges supply even a few labelled examples, or a policy table, the per-tier thresholds can be refit in seconds.
- If the hidden set might carry different caps, the system cannot infer them from text alone. The safest posture: abstain when the amount is near a learned cap (an uncertainty band) or when the amount or tier cannot be parsed, rather than guessing.
- Quantity-aware tokenisation: feed the encoder a canonicalised amount token (e.g. "<AMT_BUCKET_7>" or a scaled numeric side-feature) *in addition to* the raw text. The encoder then sees consistent number representations across currency and format styles, and the numeric comparison stays outside the transformer.
- Remove the "month = 30 days" assumption from the decision path. If durations matter (e.g. days since purchase vs a return window), parse them into days with explicit unit handling, and treat unparsed durations as abstain triggers.

### Gaps
- I found no study that quantifies quantity-aware tokenisation or numeric side-features for small encoders on policy-threshold tasks.

---

## 5. Estimating shift robustness without the hidden set

### Takeaway
Random dev splits from the same generator overstate real performance; contrast sets show drops of up to 25%. Build several *shifted* validation sets and choose model, checkpoint, WiSE α, temperature and abstention thresholds on those, not on the i.i.d. dev set: leave-one-template-out, paraphrased dev, cap-shifted and format-shifted approve cases, CheckList-style invariance and directional tests, and public-data proxies.

### Cited Findings
- CheckList: a capability × test-type matrix and a tool for generating many tests. Users found about 3× as many bugs. — [arXiv 2005.04118](https://arxiv.org/abs/2005.04118) [fetched]. (From the paper body, not re-fetched: test types are Minimum Functionality Tests, INVariance tests and DIRectional expectation tests.)
- Contrast sets: drops up to 25% across 10 datasets. — [arXiv 2004.02709](https://arxiv.org/abs/2004.02709) [fetched]
- Hendrycks et al.: "IID accuracy is not perfectly predictive of OOD accuracy, so increasing IID benchmark performance does not necessarily yield superior OOD generalization" (Table 4 caption). — [arXiv 2004.06100](https://arxiv.org/pdf/2004.06100) [fetched]

### Inferences
- **Leave-one-template-out (LOTO):** cluster the 6,500 tickets by template skeleton (mask numbers and entities, then n-gram or MinHash cluster), hold out whole clusters, and report the ID-vs-LOTO gap per task. That gap is the best available proxy for "new templates".
- **Paraphrase dev:** LLM-rewrite the 1,500 dev tickets, label-preserving, into new registers, and score the model on them.
- **Approve stress sets:** amount-format variants (INV: the label must not change), amounts at cap ± ε (DIR), unseen tiers, and scaled caps (e.g. all caps ×1.5) to measure how badly a cap change hurts. This tells you whether to abstain near caps.
- **Grounded stress sets:** KB articles reformatted (bullets vs prose, longer articles, distractor paragraphs), drafts with a single changed number or negation (contrast pairs).
- **Route proxies:** map Bitext/BANKING77 intents onto the 5 queues and evaluate zero-shot, as a "real-ish phrasing" test.
- Select calibration (temperature) and abstention thresholds on the *union* of shifted sets, so the coverage/risk curve stays honest under shift. Report the dev score alongside a "shifted" score.

### Gaps
- I found no public benchmark of template-generated to real support-ticket shift with which to calibrate the expected size of the drop.

---

## 6. Industry data-centric practice for support-ticket models in production

### Takeaway
I found no primary industry sources in this session. The standard practice described below is general MLOps knowledge, not sourced.

### Cited Findings
- None retrieved in this session. Bitext's own description of curating synthetic data with linguists at each step is the only industry-side process evidence found. — [HF Bitext card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) [fetched]

### Inferences
- For the demo or pitch, show: monitoring of input-embedding drift and confidence (MSP) drift; tracking the abstention rate as an early-warning signal; routing abstained items to humans and feeding their labels back (active learning on low-confidence and high-OOD-score items); periodic recalibration and EM prior re-estimation on recent traffic.

### Gaps
- No sourced industry case studies (e.g. Zendesk, Intercom, Salesforce) on ticket-classifier drift were found. The report writer should not present specific industry numbers.
