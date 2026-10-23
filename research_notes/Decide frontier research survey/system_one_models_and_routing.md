# System One typed-decision models (Jev, Laya) and support-ticket routing / LLM-router research

Research date: 2026-10-24. About 17 tool calls. Primary sources are the TypeSafe docs and blog, the Laya HF model card, arXiv abstracts, Salesforce Trailhead and the GLiClass HF card. Secondary sources are flowtivity, metatext, eesel and usefini. Some items come only from search-engine summaries and are flagged as such.

## Q1. Jev (TypeSafe AI): what it is, API, calibration, claims vs. verifiable facts

### Takeaway
Jev is a proprietary hosted "System One" model. It takes text or JSON state plus typed questions (Choice / Score / Noul) and returns typed values with probabilities from `POST /v1/systemone`, at $0.042 per million input tokens. TypeSafe discloses no architecture, size or calibration metrics. The "can't hallucinate" claim only means the output always fits the schema. The only independent accuracy and ECE numbers come from third parties, and they disagree on calibration.

### Cited Findings
**Verifiable product facts (official docs)**
- Jev is "TypeSafe's flagship model and the first System One model". System One models "make fast, structured decisions for software", returning typed answers and probabilities instead of generated text. — [TypeSafe docs: System One](https://docs.typesafe.ai/concepts/system-one)
- Input is text only: strings, JSON objects and arrays of text. Images, audio and video are "not supported (yet)". — [TypeSafe docs: System One](https://docs.typesafe.ai/concepts/system-one); [TypeSafe docs: Models](https://docs.typesafe.ai/models)
- It has three primitives. **Choice** picks one option from a set (e.g. `choice: "billing"`). **Score** returns a position on a defined scale (e.g. `score: 1.4` on 0–2). **Noul** gives true/false as a probability (e.g. `noul: 0.95`). Some types also return a confidence, and each type's page says which. — [TypeSafe docs: System One](https://docs.typesafe.ai/concepts/system-one)
- The docs' own worked example is a support case. Build a state from the customer message, the transactions and the refund policy. Ask independent questions: was a refund requested, is a duplicate charge indicated, does policy support a refund. Then "combine the answers with deterministic code checks and route the case." — [TypeSafe docs: System One](https://docs.typesafe.ai/concepts/system-one)
- Calibration semantics: probabilities are "optimized against outcomes". Calibration "is measured across groups of predictions" and does not guarantee that any single answer is correct. — [TypeSafe docs: System One](https://docs.typesafe.ai/concepts/system-one)
- Versioning: `jev-1.13.0` is stable and `jev-1.14.0` has been available since 2026-10-14. Alias `jev-latest` points to the latest stable release (1.13.0, the SDK default). Alias `jev-preview` points to the newest release (1.14.0). Every response's `model` field carries the versioned ID that answered, and TypeSafe recommends pinning a versioned ID for stable results. `GET /v1/models` lists only the aliases. — [TypeSafe docs: Models](https://docs.typesafe.ai/models)
- Limits: 64k tokens per request (state plus all questions) and 32k per question by default. The header `TypeSafe-Danger-Zone-Long-Context: true` raises both to 262k, at the same price but "slower and less accurate". There is no cap on the number of questions. Rate limits are set per organization in tokens/s and requests/s, and exceeding them returns 429 with `retry-after`. — [TypeSafe docs: Models](https://docs.typesafe.ai/models)
- Pricing: $42 per billion input tokens ($0.042/Mtok). "Output tokens are free." — [TypeSafe docs: Models](https://docs.typesafe.ai/models); [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**Marketing claims (launch blog, unsupported by data)**
- Trained with "Reinforcement Learning for Calibrated Decisions (RLCD)", using a "new model architecture and a parallel sampler" that produces all outputs in one query. The blog gives no parameter count, architecture details or training data. — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- The blog says every output carries "calibrated probabilities and confidence scores" and that "higher confidence means higher accuracy", but it shows no ECE figures or reliability curves. — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- The blog says Jev "can't hallucinate" and that type errors are "mathematically impossible". The post itself admits its hallucination figure "is not empirical". — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  - A secondary analysis reads the claim narrowly as a schema guarantee: Jev cannot return malformed values or undeclared options, which says nothing about correctness. — [search summary citing sanity.io / truefoundry / width.ai explainers](https://www.truefoundry.com/blog/typesafe-ai-jev)
- Speed and cost claims:
  - The blog claims 70–500 ms end-to-end, against "3 to 329 seconds" for frontier LLMs, and "40x–200x faster". The evals were run from the team's laptops on the US West Coast. — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  - The homepage claims "193.6x faster" and "444.6x cheaper". — [TypeSafe homepage via blog fetch](https://typesafe.ai/)
- The blog's workflow evals claim Jev "owns the Pareto frontier". The reference answers are the average of GPT-6 Astra and Fable 5.1, which the authors admit may bias the comparison, and no accuracy numbers are given. Jev supports option cardinality up to 255 (Wikiracing demo). — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Dates and funding:
  - The article header says Sep 15, 2026, while the page metadata says Oct 19, 2026. — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  - A $40M seed led by DCVC appears only in a search-engine summary of explainer pages, not in the blog. — [sanity.io glossary (via search)](https://www.sanity.io/glossary/jev-typesafe-ai-model)
  - The founder is Diogo Almeida, who says he worked on OpenAI instruction-following. This is self-reported. — [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**Third-party measurements of Jev**
- Metatext's typed-decisions leaderboard runs its own harness, zero-shot, with latency measured as p50 per 3-question customer-service case on **four CPU cores** (an API round trip for Jev). Jev ranks #1:
  - accuracy 0.726, beating the baseline on 18/20 tasks
  - ECE 0.079 raw and 0.049 after held-out temperature fitting
  - p50 latency 405 ms

  — [Metatext leaderboard](https://metatext.io/system-one/typed-decisions-benchmark)
- The Laya model card quotes "TypeSafe Jev 1.13.0 (third-party published)" on typed-decisions: accuracy 0.727, soft accuracy 0.580, Brier 0.148, ECE 0.144, score MAE 0.391. On other tasks it reports AG News 0.910, DAIR Emotion 0.480 and Banking77 0.870 (72 labels), with p50 latency of 236–276 ms for one question. — [HF: convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- Flowtivity reports Jev's ECE as 0.246 and says 10 questions take about 1,500 ms when sent serially. — [Flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/). **This conflicts** with the Laya card (0.144) and Metatext (0.079 raw).
- An arXiv study ("Jev in the Wild", 2609.30216, 24 Sep 2026) looks at 2,170 public GitHub projects using Jev. "Attribute judgment and scoring" are the most common uses, and routing agents draw the most public attention. It reports no accuracy or latency figures. — [arXiv 2609.30216](https://arxiv.org/abs/2609.30216)

### Inferences
- Jev's API is a strong model for Decide's interface: typed primitives, per-question criteria, versioned model IDs echoed in every response, and an explicit statement of what calibration means. Decide's three questions map onto two primitives. route is a Choice with 5 options; approve and grounded are both Noul.
- Jev cannot meet a p95 < 25 ms on-box budget: 236–405 ms per API round trip. A 17M on-CPU encoder that runs about 10–20x faster than Jev is a real differentiator worth stating in the pitch.
- Our own "never hallucinates" wording should be the narrow, honest version: outputs are schema-closed, and abstention is explicit.

### Gaps
- No official Jev architecture, parameter count, training data, or published ECE / reliability curves.
- The Confidence and Primitives doc pages were not fetched, so the exact response JSON fields (probability vs. confidence) are unconfirmed.
- The Jev ECE conflict (0.049–0.246) cannot be resolved. The harnesses and temperature-fitting protocols differ.

## Q2. Laya (convaiinnovations/laya): model card details and independent benchmarks

### Takeaway
Laya is an open (Apache-2.0) re-implementation of the Jev idea on ModernBERT-large (421M). Each option is scored at its own [MASK] token in one forward pass. It was trained with RL against proper scoring rules and still ships over-confident, so the card tells users to refit a temperature. Its strong results come from in-domain fine-tuning. Zero-shot it sits near chance, and on CPU it is far too slow for a 25 ms budget.

### Cited Findings
**Checkpoints, architecture and API**
- Laya is a "System 1 decision model": non-autoregressive, it "never generates text", and it takes state (text, email dict or JSON) plus typed questions, returning typed answers with probabilities in one forward pass. License Apache-2.0. — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Three checkpoints:
  - root: ModernBERT-large, 421M parameters, 512 context, English only
  - `laya-multilingual`: mmBERT-base, 322M parameters, 1024 context (up to 8192), 100+ languages
  - `laya-typed-decisions`: ModernBERT-large, 421M parameters, 1024 context, fine-tuned on four typed-decision workflows

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Architecture:
  - A fully fine-tuned backbone plus a decision head trained from scratch: 2 transformer layers, an option-marker scorer and an act/escalate head.
  - "Each option is scored at its own [MASK] token, then softmaxed across that question's options." New schemas need no retraining.
  - The question/options take up `head_max_len` = 192 tokens (English) or 256 (multilingual); the rest of the context holds the state.

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Input and output format:
  - Questions are a dict keyed by name. Each has `type` (`choice` | `score` | `noul`), `instructions` and `criteria` (option descriptions for a choice, ordered levels for a score).
  - The output is `result["answers"][name][type]`, plus `result["routing"]` giving the `model`, `repo` and `reason`, because a language router picks the checkpoint.

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)

**Training and calibration**
- Training is "RLCD":
  - The policy outputs a distribution, exploration adds zero-mean Gaussian noise to the logits, and the reward is a strictly proper scoring rule (log + spherical, plus RPS for ordinal questions).
  - Updates use REINFORCE with a group-mean baseline, and multi-turn conversations use TD(λ=1).
  - The training corpora are not specified.

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Calibration as shipped: the models are over-confident. The card recommends refitting **one temperature per (question type, option count)** on your own data. Mean ECE drops from 0.466 to 0.081 (laya) and from 0.314 to 0.106 (multilingual). — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Abstention: the act/escalate head is broken. `act_probability` reads about 1.0 for almost every input, with AUROC 0.30 on 396 labelled decisions. The card says to gate on `confidence` instead, which reaches AUROC 0.77. — [HF: laya](https://huggingface.co/convaiinnovations/laya)

**Reported metrics**
- typed-decisions benchmark (accuracy / soft accuracy / Brier / ECE / score MAE):
  - laya-typed-decisions: 0.766 / 0.471 / 0.062 / 0.213 / 0.242
  - laya: 0.362 / 0.332 / 0.316 / 0.175 / 0.694
  - laya-multilingual: 0.342 / 0.326 / 0.439 / 0.285 / 0.687
  - Reference points: teacher self-agreement ceiling 0.735, majority class 0.461, random 0.318

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Per-workflow results (fine-tuned checkpoint):
  - invoice processing 0.804, security incidents 0.766, **customer service 0.764**, agent-trace observability 0.730
  - by primitive: noul 0.857, choice 0.733, score 0.723

  — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Other tasks: MASSIVE intent (English) 0.783, XNLI (English) 0.860, AG News 0.950, DAIR Emotion 0.595, **Banking77 0.425**. The card explains the Banking77 score: each of the 77 labels gets only about 3–4 tokens of head budget, and the suggested fixes are a larger `head_max_len` or a hierarchical choice. SST-5 (ordinal) scores 0.372. — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Speed on a T4 GPU (laya / multilingual): 1 question 39.5 / 32.8 ms, 5 questions 84.5 / 40.1 ms, 10 questions 158.6 / 72.3 ms, 50 questions 771 / 337 ms. — [HF: laya](https://huggingface.co/convaiinnovations/laya)

**Known failure modes (from the card)**
- A `noul` answer may follow its `false:`/`true:` option labels rather than the state (issue #156). The workaround is a 2-option choice with neutral keys. — [HF: laya](https://huggingface.co/convaiinnovations/laya)

**Independent findings**
- Flowtivity ("benchmarked honestly"):
  - Its own CPU test (4 vCPU, 7 GB RAM, Laya 0.3.4): import 26 s, checkpoint load 675 s including an 808 MB download, first predict 85.4 s, **warm median 49.4 s**. The documented CPU figure is 193–464 ms on better hardware. Its conclusion: without a GPU, Laya is "a batch tool, not a real-time request path."
  - Failures it found:
    - Khmer: 0.000 accuracy at 0.952 confidence on the English checkpoint.
    - Choice accuracy degrades past about 20 options.
    - The 512-token window holds about one email.
    - Its outputs drifted from the published quickstart (churn risk 19.3% vs. 91.4%).
  - Specialist checkpoints: spam 0.993, phishing 0.980, jailbreak 0.755–0.762, and **0.931 accuracy at 50% selective coverage**.

  — [Flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
- Metatext (one harness, 4 CPU cores):
  - LAYA typed-decisions: accuracy "0.714 → 0.762" (the meaning of the arrow is unexplained, and the run is in-domain because the model was fine-tuned on the train split), ECE 0.161 raw → 0.016 held-out, **p50 3.5 s**.
  - LAYA base: 0.343, 3.2 s. LAYA multilingual: 0.345, 1.2 s.
  - Dataset: synthetic, 4 workflows × 100 JSON states × 5 questions = 2,000 decisions. Gold labels are the argmax of a teacher distribution.

  — [Metatext leaderboard](https://metatext.io/system-one/typed-decisions-benchmark)
- Not fetched: innfactory, ai-tldr, mer.vin and i-scoop pages exist ("Laya: The 33ms Open-Source Decision Model Beating Jev"). Their headlines repeat vendor numbers. — [innfactory](https://innfactory.ai/en/ai-models/laya/); [ai-tldr](https://ai-tldr.dev/releases/convai-laya/); [mer.vin](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev/)

### Inferences
- Laya's design is very close to Decide's: typed prompts, a shared encoder, per-option scoring, post-hoc temperature per type, and gating on confidence. That validates our approach. Our edges to claim are about 25x fewer parameters (17M vs. 421M), CPU p95 < 25 ms (Laya runs 193 ms to seconds on CPU), and a *working* abstain signal. Report the AUROC of our confidence for selective prediction, as Laya does (0.77), plus accuracy at fixed coverage (Laya specialist: 0.931 at 50%).
- Robustness lesson: Laya's zero-shot checkpoints fall below the majority baseline, while fine-tuned in-domain ones beat the teacher ceiling. If judges swap the dataset, a model fine-tuned on fixed label names will degrade the same way. Mitigations:
  - Put label descriptions (criteria) into the input, Laya-style, so label semantics are read rather than memorized.
  - Refit per-question temperature and thresholds on any labelled sample the judges provide.
  - Fall back to abstaining when the input looks out of distribution (Flowtivity's Khmer case shows confident failures under shift).
- Copy Laya's calibration granularity: one temperature per (question type, option count). For Decide that is three temperatures: route/5, approve/2, grounded/2.
- Its `noul` label-bias bug argues for testing our approve and grounded heads on label-flipped or neutral-key prompts.

### Gaps
- Laya's training corpora and the size of the RLCD teacher are undisclosed.
- No independent CPU latency for Laya on hardware like ours, beyond Flowtivity's anomalous 49 s and Metatext's 3.5 s.

## Q3. Comparable single-forward-pass zero-shot classifiers / decision models

### Takeaway
GLiClass, GLiNER2 (Fastino) and NLI cross-encoders (deberta-v3-large-zeroshot) are the main open alternatives. On the typed-decisions suite all of them score far below Jev and fine-tuned Laya (0.40–0.49 vs. about 0.73), though after temperature fitting they calibrate well (ECE 0.03–0.09).

### Cited Findings
- Metatext zero-shot typed-decisions results (accuracy / ECE raw → held-out / p50 on 4 CPU cores):

  | Model | Accuracy | ECE raw → held-out | p50 |
  |---|---|---|---|
  | GLiNER2 large (Fastino) | 0.486 | 0.300 → 0.029 | 3.8 s |
  | Von (wfzyx) | 0.461 | 0.249 → 0.027 | 597 ms |
  | GLiClass large v3.0 | 0.446 | 0.315 → 0.093 | 977 ms |
  | GLiNER 2.5 base | 0.398 | 0.415 → 0.059 | 1.3 s |
  | GLiClass multilang ultra | 0.395 | 0.281 → 0.031 | 672 ms |

  — [Metatext](https://metatext.io/system-one/typed-decisions-benchmark)
- GLiClass-large-v3.0:
  - About 439M parameters, GLiNER-inspired. It scores all labels together in a "single forward path" and claims cross-encoder accuracy at lower compute.
  - Zero-shot average F1 is 0.7193 over 13 datasets (the card's table shows 0.7001, which is inconsistent). Highs: snips 0.969, sst2 0.919. Lows: sst5 0.461, emotion 0.451.
  - The best cross-encoder baseline, deberta-v3-large-zeroshot-v2.0, averages 0.6821.
  - Speed on an A6000 at batch size 1: about 25 examples/s for large, 51.6 for base, 97.3 for edge, against 6.0 for deberta-v3-large.
  - Apache-2.0; paper arXiv 2508.07662.

  — [HF: knowledgator/gliclass-large-v3.0](https://huggingface.co/knowledgator/gliclass-large-v3.0)

### Inferences
- A cheap way to test robustness to a new dataset is to build a zero-shot fallback ensemble: our model plus a GLiClass-edge-style label-description scorer. But GLiClass's CPU latency (hundreds of ms for the large model) rules it out of the 25 ms hot path, unless the edge variant is distilled into ours.
- Every zero-shot model calibrates to ECE ≤ 0.09 after held-out temperature fitting. Post-hoc temperature is the universal fix, and calibration alone is no differentiator. Selective-prediction quality (AUROC, risk–coverage curves) is.

### Gaps
- Llama Guard and Granite Guardian as classifiers, tasksource/deberta, and Fastino's own GLiNER2 claims were not researched for lack of budget.

## Q4. Industry support triage: Zendesk, Intercom, Salesforce, Ada, Forethought, Freshdesk

### Takeaway
The industry standard is tiered automation by confidence: recommend, then pre-fill for agent confirmation, then auto-apply. A low-confidence result falls back to a human or a default queue. Thresholds are set per field, and teams run in shadow/test mode first. Vendors publish resolution-rate marketing figures, not routing accuracy or calibration figures.

### Cited Findings
- **Salesforce Einstein Case Classification** has three automation levels per field:
  - lowest: recommend the top 3 values
  - "Select Best Value": pre-select for the agent to confirm, with a minimum-confidence slider
  - "Automate Value": save the value without review

  The Automate threshold must be higher than the Select Best Value threshold, and a prediction below a tier's threshold "falls back to the next less automated option." Auto-saving requires the paid version, and Case Wrap-Up supports only Select Best Value. — [Trailhead: Einstein Classification](https://trailhead.salesforce.com/content/learn/modules/service_case_class/learn-about-einstein-case-classification); [Trailhead: Turn on](https://trailhead.salesforce.com/en/content/learn/modules/service_case_class/turn-on-einstein-case-classification)
- **Zendesk Intelligent Triage**
  - It predicts intent, sentiment and language (about 150 languages for language detection; intent and sentiment only for a listed subset). — [Zendesk help](https://support.zendesk.com/hc/en-us/articles/4550640560538-Automatically-detecting-customer-intent-sentiment-and-language)
  - "Intent confidence" is a reportable ticket attribute in Explore. — [Zendesk Explore recipe](https://support.zendesk.com/hc/en-us/articles/5429030095258-Explore-recipe-Comparing-intelligent-triage-intent-predictions-with-a-custom-About-field)
  - Third-party, unverified: the AI agent's default intent confidence threshold is 60%, with a "50–70% sweet spot", and anything below it triggers a fallback or escalation. — [eesel](https://www.eesel.ai/blog/zendesk-ai-agent-intent-confidence-threshold)
  - Setup practice: run triage in Test Mode alongside existing routing, use the Triage Performance dashboard, and add a misroute-feedback button. — [eesel setup guide](https://eesel.ai/blog/zendesk-intelligent-triage-a-complete-setup-and-automation-guide)
- **Intercom Fin** (resolution, not routing):
  - 51% average resolution rate "out of the box", up to 86% in some deployments, per an Intercom case study on Anthropic's site (search summary; not fetched). — [Anthropic: Intercom](https://www.anthropic.com/customers/intercom)
  - A G2 listing says 67% average, up to 93% (vendor-supplied). — [G2](https://ai.g2.com/marketplace/tools/fin-by-intercom)
- **Forethought Triage** (claims via a competitor, Fini):
  - "30% reduction in handle time, 50% improvement in routing accuracy" at Upwork and Carta.
  - Needs 30–90 days of supervised tuning.

  Unverified. — [usefini](https://www.usefini.com/guides/leading-ai-ticket-triage-vendors-enterprise-support)
- **Ada**: no published routing accuracy figure was found. — [usefini](https://www.usefini.com/guides/leading-ai-ticket-triage-vendors-enterprise-support)

### Inferences
- Map Decide's output onto Salesforce's three tiers. Below a lower threshold, abstain and send to a human. Between the two thresholds, suggest the answer. Above the upper threshold, auto-act. For approve, apply the 3:1 cost asymmetry: the Bayes-optimal auto-approve threshold is p(correct to approve) > 3/(3+1) = 0.75 before any abstain margin. This framing is industry-recognizable to judges.
- Shadow mode, misroute feedback and confidence reported as a field are standard practice. A startup-grade Decide would log every decision with its confidence and model version to support recalibration.

### Gaps
- No official numbers on routing accuracy or calibration from Zendesk, Intercom, Ada, Forethought or Freshdesk Freddy. Freshdesk was not researched.
- Public intent benchmarks (BANKING77, CLINC150) were not fetched, so no state-of-the-art numbers are cited. The only Banking77 points here are Jev 0.870 and Laya 0.425 from the Laya card.

## Q5. LLM-router research (RouteLLM, RouterBench, Martian, NotDiamond)

### Takeaway
LLM routers are binary or multi-way classifiers that predict win probability, combined with a user-set threshold. The threshold is calibrated on sample traffic to hit a target fraction routed to the expensive path. That transfers directly to Decide: calibrate abstain/auto-act thresholds to a target coverage or cost, and expect drift under traffic shift.

### Cited Findings
- RouteLLM (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica; arXiv 2406.18665, June 2024, v4 Feb 2025): routers trained on preference data choose between a strong and a weak LLM, cutting costs "by over 2 times in certain cases" without lowering quality. — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- How RouteLLM thresholds work (via search summary of the repo):
  - The router estimates the strong model's win rate, and a request goes to the strong model if that estimate exceeds a threshold in [0, 1].
  - Thresholds are calibrated from sample queries to a target share of strong-model calls (example: 0.11593 for about 50% on Chatbot Arena data).
  - Calibrated thresholds are dataset-specific and shift with live traffic.

  — [GitHub lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
- RouterBench (Hu, Bieker, Li, Jiang, Keigwin, Ranganath, Keutzer, Upadhyay; arXiv 2403.12031, Mar 2024; Martian-affiliated repo withmartian/routerbench): more than 405k inference outcomes, a theoretical framework for routing, and a cost–quality evaluation. — [arXiv 2403.12031](https://arxiv.org/abs/2403.12031)
- A Research Square preprint studies cost-constrained routing with audit calls and "certified thresholds under error and harm constraints" across five RouterBench traffic settings (search summary only). — [Research Square rs-11299408](https://www.researchsquare.com/article/rs-11299408)

### Inferences
- What transfers:
  1. Choose thresholds by a target operating point (coverage or expected cost), not a fixed 0.5.
  2. Report cost–quality (risk–coverage) curves the way RouterBench does.
  3. Recalibrate thresholds whenever the traffic distribution changes. This is directly relevant if judges use a different dataset.
  4. Certified or conformal thresholds under an error budget are a research-grade upgrade for the 3:1 approve task: guarantee wrong-approval rate ≤ α at maximum coverage.

### Gaps
- NotDiamond and Martian product docs were not fetched. RouteLLM per-router results (matrix factorization, BERT, causal LLM, SW ranking) and the APGR metrics were not retrieved.

## Q6. What a startup-grade "typed question" API should include

### Takeaway
Combining Jev's and Laya's designs, the API should cover:
- typed primitives with declared option criteria
- per-request named questions answered in one pass
- versioned model IDs plus moving aliases, with the answering version echoed in each response
- documented calibration semantics (group-level, not per-answer)
- an explicit confidence/abstain field
- context limits
- routing metadata with a reason

### Cited Findings
- Jev: one endpoint `POST /v1/systemone` with a `model` field. Pinned versions (`jev-1.13.0`) sit alongside `jev-latest` / `jev-preview` aliases, and each response returns the versioned ID. The docs state per-request and per-question token limits, rate-limit 429 handling with `retry-after`, and an opt-in long-context header described as "less accurate". — [TypeSafe Models](https://docs.typesafe.ai/models)
- Jev: primitives Choice / Score / Noul. Some types return confidence. Calibration is defined across groups of predictions, and answers are meant to be combined with deterministic code checks. — [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one)
- Laya: questions are a dict of {type, instructions, criteria}, so new schemas need no retraining. The response includes `routing` {model, repo, reason}, and the card publishes a calibration recipe and an honest note that the escalate head carries no signal. — [HF: laya](https://huggingface.co/convaiinnovations/laya)
- Salesforce: per-field confidence tiers (recommend, select, automate) with ordered thresholds. — [Trailhead](https://trailhead.salesforce.com/en/content/learn/modules/service_case_class/turn-on-einstein-case-classification)

### Inferences
- A suggested Decide response schema: `{model: "decide-1.0.0", question, type, answer, probabilities{option: p}, confidence, decision: act|suggest|abstain, threshold_version, calibration{method:"temperature", fitted_on, ece}, latency_ms}`.
- For new decision types, accept `criteria` (label descriptions) at request time. Changing the label set then needs only a recalibration, not retraining. This doubles as the dataset-shift defense.
- Publish a model card with the same honesty level Laya uses: ECE raw vs. calibrated, AUROC of confidence for abstention, accuracy at fixed coverage, CPU latency p50/p95, and known failure modes.

### Gaps
- TypeSafe's Confidence and Primitives pages (exact confidence semantics and response JSON) were not fetched.
