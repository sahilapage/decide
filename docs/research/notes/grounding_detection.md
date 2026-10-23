# Grounding / hallucination detection / factual-consistency checking with small models (for Decide's "grounded" decision: is the drafted support reply supported by the KB article?)

Scope note: research done 2026-10-24 over primary sources (arXiv, HF model cards, GitHub READMEs, LLM-AggreFact leaderboard). Some figures are vendor/self-reported and flagged as such. About 21 tool calls; anything not verified is listed under Gaps.

---

## 1. Models: which small grounding detectors exist, their sizes, licences, benchmark scores and CPU latency?

### Takeaway
The strongest *small* (≤0.8B) open checkers on LLM-AggreFact are FactCG-DeBERTa-L (0.4B, 75.6–77.2 BAcc), MiniCheck-Flan-T5-L (0.77B, ~75.0) and MiniCheck-RoBERTa/DeBERTa-L (0.355B, ~72.6–74.4). They come within about 1–2 points of GPT-4o (75.9) and Bespoke-MiniCheck-7B (77.4). The most directly relevant model for Decide is **TinyLettuce-17M**. It uses the *same Ettin-17M backbone* and is MIT-licensed, with RAGTruth example-level F1 of 68.5, against 79.2 for LettuceDetect-large (395M). That gap is roughly what a 17M encoder costs on real-world RAG hallucinations compared with larger encoders.

### Cited Findings

**LLM-AggreFact leaderboard (11 grounded-generation datasets; the page shows 14 of 42 models; average BAcc):**
- nimble-v4-max (27B) 79.5; Jev 1.1.3.0 (size n/a) 78.7; nimble-v4 (9B) 77.8; **Bespoke-MiniCheck-7B 77.4**; Claude-3.5 Sonnet 77.2; **Granite Guardian 3.3 (8B) 76.5**; Mistral-Large 2 76.5; gpt-4o-2024-05-13 75.9; **FactCG-DeBERTa-L (0.4B) 75.6**; Qwen2.5-72B 75.6; **MiniCheck-Flan-T5-L (0.8B) 75.0**; Llama-3.3-70B 74.5; Llama-3.1-405B 74.4; QwQ-32B 71.8. Per-dataset for the small models: FactCG-DeBERTa-L scores CNN 70.1, XSum 73.9, MediaS 72.3, MeetB 74.3, WiCE 74.2, REVEAL 88.4, ClaimVerify 78.5, FactCheck 72.1, ExpertQA 59.1, LFQA 86.7, RAGTruth 82.3. MiniCheck-FT5 scores RAGTruth 78.0, LFQA 85.2, ExpertQA 59.0. — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)
- ExpertQA sits at about 59–61 for *every* model, including Claude-3.5 and GPT-4o. Some subsets are close to label-noise-limited. — [LLM-AggreFact leaderboard](https://llm-aggrefact.github.io/)

**MiniCheck (Tang, Laban, Durrett, EMNLP 2024; arXiv 2404.10774):**
- Sizes and LLM-AggreFact BAcc at a fixed threshold of 0.5 (paper): MiniCheck-RBTA (RoBERTa-L) 355M → 72.7; MiniCheck-DBTA (DeBERTa-v3-L) 355M → 72.6; MiniCheck-FT5 (Flan-T5-L) 770M → 74.7; AlignScore 355M → 70.4; SummaC-ZS 60M → 67.9; SummaC-CV 60M → 62.1; QAFactEval 1.4B → 66.5; T5-NLI-Mixed 11B → 61.0; GPT-4 → 75.3. — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- Cost on a 13K test set at $0.8/GPU-hr: MiniCheck-RBTA/DBTA about $0.20 and FT5 about $0.24, against $107 for GPT-4 ($161–212 with decontextualisation or decomposition). The authors call FT5 "more than 400× cheaper than GPT-4". — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- Repo: Apache-2.0 code. Model names `roberta-large`, `deberta-v3-large`, `flan-t5-large`, `Bespoke-MiniCheck-7B`. Bespoke-MiniCheck-7B is "current SOTA regardless of size", but commercial use requires contacting company@bespokelabs.ai. It needs vLLM (Linux) and is on Ollama as `bespoke-minicheck`. On one A6000 the 29K-example test set took about 30 min with prefix caching and 55 min without. — [MiniCheck GitHub](https://github.com/Liyan06/MiniCheck)

**FactCG (Lei et al., Microsoft, NAACL 2025; arXiv 2501.17144):**
- FactCG-RBT (RoBERTa-L, ~0.4B), FactCG-DBT (DeBERTa-v3-L, ~0.4B), FactCG-FT5 (Flan-T5-L, ~0.8B).
- LLM-AggreFact BAcc with per-dataset tuned thresholds: GPT-4o 75.9; AlignScore 72.5; MiniCheck-RBT 74.4 / DBT 74.1 / FT5 75.5; FactCG-RBT 75.4 / **DBT 77.2** / FT5 76.7.
- At a fixed 0.5 threshold: FactCG-RBT 73.4 / DBT 75.6 / FT5 75.0; MiniCheck-RBT 73.5 / DBT 73.1 / FT5 75.0; AlignScore 70.5.
- The paper reports FactCG-DBT's gain over the MiniCheck models as significant (p<0.05, paired bootstrap), from a single run per benchmark.
- Code is at github.com/derenlei/FactCG. The paper states no licence for FactCG itself. — [FactCG paper](https://arxiv.org/html/2501.17144)

**AlignScore (Zha et al., ACL 2023; arXiv 2305.16739):**
- 355M parameters (RoBERTa-L). Trained on 4.7M examples from 7 tasks: NLI, QA, paraphrase, fact verification, IR, semantic similarity and summarisation.
- Evaluated on 22 datasets, 19 unseen in training. The authors claim it "matches or even outperforms" ChatGPT/GPT-4-based metrics. — [AlignScore abstract](https://arxiv.org/abs/2305.16739)
- AlignScore code dependency licence is MIT, as listed in FactCG's licence table. — [FactCG paper](https://arxiv.org/html/2501.17144)

**Vectara HHEM-2.1-Open (`vectara/hallucination_evaluation_model`):**
- Fine-tuned from google/flan-t5-base. The card lists 0.1B params (F32). Apache-2.0. Input is (premise, hypothesis) pairs, with output 0–1 for consistent vs hallucinated.
- BAcc / F1: AggreFact-SOTA 76.55 / 66.77 (GPT-4 73.78 / 63.87); RAGTruth-Summ 64.42 / 44.83 (GPT-4 62.62 / 40.59); RAGTruth-QA 74.28 / 60.00 (GPT-4 74.11 / 57.78). HHEM-1.0 scored 78.87 on AggreFact-SOTA, slightly higher.
- Uses under 600MB RAM at fp32 and takes **about 1.5 s for a 2k-token input on a modern x86 CPU**. Claimed "unlimited" context (HHEM-1.0 was 512).
- HHEM-2.3 (11 languages) is available only through Vectara's platform. — [HHEM model card](https://huggingface.co/vectara/hallucination_evaluation_model)

**LettuceDetect (KRLabs, arXiv 2502.17125), token-level, ModernBERT:**
- lettucedetect-base-v1 is about 150M and large-v1 about 396M (ModernBERT). It is a token classifier over [context, question, answer]. Tokens above 0.5 that run consecutively form hallucinated spans. Max input 4,096 tokens in v1.
- RAGTruth example-level F1: large 79.22 (P 80.44 / R 78.05), base 76.07. Comparisons: RAG-HAT (Llama-3-8B FT) 83.9; Llama-2-13B FT 78.7; **Luna 65.4**; gpt-4-turbo prompt 63.4; Trulens 60.4; SelfCheckGPT 58.8; RAGAS Faithfulness 52.0.
- Span-level F1: large 58.93, base 55.44, Llama-2-13B FT 52.7, gpt-4-turbo 28.3.
- Speed is about 30–60 examples/s on a single GPU (GPU unnamed). — [LettuceDetect paper](https://arxiv.org/html/2502.17125v1)
- Code and models are MIT. Later variants: EuroBERT 210M/610M (8K context, multilingual), lettucedect-v2-mmbert-base (span-F1 0.642 on a unified 10,698-example v2 test set, 0.528 on the RAGTruth slice) and lettucedect-v2-qwen-2b (0.689 / 0.574). — [LettuceDetect README](https://cdn.jsdelivr.net/gh/KRLabsOrg/LettuceDetect@main/README.md)

**TinyLettuce (Ettin 17M/32M/68M; added in v0.1.8, 2025-08-31):**
- HF: `KRLabsOrg/tinylettuce-ettin-17m-en-v1`, base `jhu-clsp/ettin-encoder-17m`. 8K context (trained at max length 4096). MIT.
- RAGTruth example-level F1: **17M 68.52**, 32M 72.15, 68M 74.97, against LettuceDetect-base (150M) 76.07 and large (395M) 79.22.
- Synthetic test set (600 examples, generated by GPT-5-mini), F1: 17M 90.87 (P 84.56 / R 98.21), 32M 88.76, 68M 92.64, base-ModernBERT 87.60, GPT-5-mini 83.69, GPT-OSS-120B 83.38, Qwen3-235B 79.84. These are self-reported, on in-distribution synthetic data.
- Training: the HallucinationGenerator injects error types (e.g. numerical, temporal) at an "intensity" (e.g. 0.4–0.5) into `enelpol/rag-mini-bioasq`, using GPT-OSS-120B as generator. Synthetic-only models used 3,000 samples (1,500 / 1,500). "General" models used RAGTruth plus 3,000 synthetic. Settings: AdamW, lr 1e-5, wd 0.01, 5 epochs, bs 16. No measured CPU latency is published. — [TINYLETTUCE.md](https://cdn.jsdelivr.net/gh/KRLabsOrg/LettuceDetect@main/docs/TINYLETTUCE.md)
- Third-party caveat: the LettucePrevent streaming benchmark reports tinylettuce-ettin-68m-en at binary F1 0.2856 on the hallucination class, below the other detectors compared. — [LettucePrevent blog (HF)](https://huggingface.co/blog/lebe1/lettuceprevent-blogpost) (search-snippet level; not fetched in full)

**Galileo Luna (arXiv 2406.00975):**
- DeBERTa-large (440M), fine-tuned for RAG hallucination detection. It uses sliding windows to handle long contexts (16K, per a secondary summary).
- Claims 97% lower cost and 96% lower latency than GPT-3.5 (v1 abstract); the ACL version says 91% latency. — [Luna arXiv](https://arxiv.org/abs/2406.00975v1); [Galileo blog](https://galileo.ai/blog/galileo-luna-breakthrough-in-llm-evaluation-beating-gpt-3-5-and-ragas)
- On RAGTruth it is clearly beaten by LettuceDetect (65.4 vs 79.2 F1). — [LettuceDetect paper](https://arxiv.org/html/2502.17125v1)

**ORION (Deepchecks, arXiv 2504.15771):**
- An encoder-based NLI model with a 512-token window, combined with retrieval over long context. It reports RAGTruth response-level F1 of 0.83 and claims to "match methods trained on the dataset".
- Parameter count, latency and weights release are not stated in the abstract. — [ORION abstract](https://arxiv.org/abs/2504.15771)

**Granite Guardian (IBM, arXiv 2412.07724):**
- Covers RAG groundedness, context relevance and answer relevance. Versions are 3.0 (2B/8B), 3.2-3B-A800M (MoE) and 3.3-8B. Apache-2.0.
- Paper AUC is 0.854 on RAG-hallucination benchmarks. Granite Guardian 3.3-8B scores 76.5 on LLM-AggreFact. — [IBM docs](https://www.ibm.com/granite/docs/models/guardian); [paper](https://arxiv.org/html/2412.07724v2); [HF 3.2-3b-a800m](https://huggingface.co/ibm-granite/granite-guardian-3.2-3b-a800m); [leaderboard](https://llm-aggrefact.github.io/)

**Lynx (Patronus, arXiv 2407.08488):**
- 8B and 70B fine-tunes of Llama-3-Instruct. Lynx-70B averages 87.4% and Lynx-8B 84.3% on HaluBench (~15k samples). Lynx 2.0 8B is claimed +2.2% over Claude-3.5-Sonnet on HaluBench. All figures are vendor-reported.
- The 70B is listed as CC-BY-NC-4.0 by a third-party host; I did not verify this on the official card. — [Patronus docs](https://docs.patronus.ai/docs/research_and_differentiators/Lynx); [Lynx paper](https://arxiv.org/html/2407.08488v2); [featherless (licence)](https://featherless.ai/models/PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct/readme)

### Inferences
- **Nothing published fits Decide's 25 ms CPU p95 budget off the shelf** except Ettin/MiniLM-scale encoders. HHEM (T5-base) needs about 1.5 s per 2k tokens on CPU. The 355M–800M checkers are 20–50× bigger than Ettin-17M. The 7B–8B models (Bespoke, Granite Guardian, Lynx) are far out of budget.
- TinyLettuce-17M is the best published evidence of what an Ettin-17M grounding model can do. It reaches about 68.5 RAGTruth F1, against 79.2 for a 395M ModernBERT, so about 10 F1 is lost going from 395M to 17M. 68M recovers most of the gap (75.0). That suggests Decide's grounding head will generalise noticeably worse than the 0.4B models on messy data, and the hand features (numbers, scope) need to carry precise slot-level errors.
- Large models can serve as **offline teachers and labellers** for distillation into the 17M student: MiniCheck-FT5, FactCG-DBT, Bespoke-MiniCheck-7B (licence permitting) or HHEM (Apache-2.0).
- Since our model shares the Ettin backbone, TinyLettuce-17M's MIT weights could be a **warm-start checkpoint** or a second-opinion feature. Its token-level formulation over (context, answer) matches "draft vs KB".

### Gaps
- No primary-source CPU latency for TinyLettuce, MiniCheck-RBTA/DBTA, FactCG or AlignScore. Only HHEM's 1.5 s / 2k tokens is published.
- The leaderboard rows for MiniCheck-RoBERTa/DeBERTa, AlignScore and HHEM were hidden in the visible view, so their current leaderboard numbers were not retrieved. Paper numbers are given above instead.
- ORION model size and release status, Lynx-8B licence, and Patronus "Glider" and other 2026 small detectors were not verified.
- I did not find FactCC, SummaC-Conv, QAFactEval or TRUE-T5 numbers beyond MiniCheck's table. These are older (2020–2022) and dominated by the models above.
- One fetched summary of the MiniCheck paper said MiniCheck-RBTA was trained on the 14K synthetic data only, while FT5/DBTA add 21K ANLI. I have not confirmed this against the paper body.

---

## 2. Benchmarks: which ones match "support reply vs KB article"?

### Takeaway
No public benchmark is customer-support-specific. The closest matches are RAG-style grounded QA sets: **RAGTruth-QA**, LLM-AggreFact's **ClaimVerify / ExpertQA / LFQA / RAGTruth** subsets, and to a lesser degree WiCE/REVEAL (claim vs evidence). The CNN/XSum/SummEdits/FRANK and MediaSum/MeetingBank sets are summarisation-oriented and less representative. FaithBench is a deliberately adversarial stress test where detectors score about 50%.

### Cited Findings
- **LLM-AggreFact** aggregates 11 human-labelled grounded-generation datasets: AggreFact-CNN/XSum, TofuEval-MediaS/MeetB, WiCE, REVEAL, ClaimVerify, FactCheck-GPT, ExpertQA, LFQA and RAGTruth. It reports balanced accuracy, and the leaderboard lists 42 models. — [LLM-AggreFact](https://llm-aggrefact.github.io/); [MiniCheck GitHub](https://github.com/Liyan06/MiniCheck)
- **RAGTruth** (ParticleMedia, MIT licence): 2,965 instances and 17,790 responses from 6 LLMs (GPT-3.5/4, Llama-2-7/13/70B, Mistral-7B). It has 7,664 hallucinated responses and 14,289 annotated spans. Tasks are QA (989; MS MARCO-style), data-to-text (1,033; Yelp), and summarisation (628 CNN/DM + 315 recent news). Span labels distinguish evident vs subtle conflict and baseless info; the README shows "Evident Baseless Info". — [RAGTruth GitHub](https://github.com/ParticleMedia/RAGTruth); [RAGTruth paper](https://arxiv.org/abs/2401.00396)
- **TRUE** (Honovich et al. 2022): standardises 11 factual-consistency datasets (summarisation, dialogue, fact verification, paraphrase) to binary consistent/inconsistent labels, evaluated by ROC-AUC. — [TRUE](https://ar5iv.labs.arxiv.org/html/2204.04991)
- **FaithBench** (Vectara, arXiv 2410.13210): summaries from 10 LLMs across 8 families, chosen where popular detectors (including GPT-4o-as-judge) disagree, with expert labels. "Even the best detectors score around 50% accuracy." — [FaithBench](https://arxiv.org/abs/2410.13210)
- **HaluBench** (Patronus): about 15k samples from real-world domains, used for Lynx. — [Patronus docs](https://docs.patronus.ai/docs/research_and_differentiators/Lynx)
- **EQUATE** tests quantitative NLI. Nine published NLI models showed essentially zero average gain over the majority baseline on quantity reasoning. The symbolic Q-REAS baseline gained +24.2% on numerical tests but lost 8.1% on verbal reasoning. — [EQUATE (ACL Anthology)](https://aclanthology.org/K19-1033/)

### Inferences
- For a judge-held-out dataset, the best *public proxy* is **RAGTruth-QA plus the ClaimVerify/LFQA/ExpertQA subsets**. A "support reply vs KB" test will likely include unsupported additions ("baseless info") and subtle conflicts, both annotated in RAGTruth. It will be multi-sentence, as LFQA is.
- EQUATE suggests learned encoders won't reliably learn number/unit reasoning from NLI data alone. That is evidence for keeping Decide's symbolic number/duration normaliser alongside any learned model, rather than replacing it.
- Report BAcc and per-subset numbers, since thresholds and base rates differ across subsets (see §4).

### Gaps
- SummEdits, FRANK and HaluEval numbers were not retrieved. They are summarisation- or QA-oriented and less directly relevant.
- I found no public customer-support/KB grounding benchmark. TofuEval (dialogue summarisation) is the closest "conversational" set.

---

## 3. Techniques that generalise: decomposition, chunking, token-level spans, synthetic hard negatives, numeric/slot checks

### Takeaway
Across the strong small checkers, the recipe that generalises has four parts. (a) **Split the claim into sentences** and check each against the evidence independently. (b) **Chunk long evidence and take the max over chunks** per sentence, then aggregate across sentences (min, or "any unsupported → unsupported"). (c) Train on **LLM-generated hard negatives** built by removing exactly the evidence needed (MiniCheck C2D/D2C, FactCG CG2C). That data is worth about 15 BAcc points over training without it. (d) Treat atomic-fact decomposition with caution: it can strip context and add noise.

### Cited Findings
- MiniCheck checks claims sentence by sentence, each without preceding context. A sentence's score is the **max over its grounding documents/chunks**, so it counts as supported if any chunk supports it. The repo recommends splitting multi-sentence claims into sentences "for best performance". — [MiniCheck paper](https://arxiv.org/html/2404.10774); [MiniCheck GitHub](https://github.com/Liyan06/MiniCheck)
- **C2D (claim→doc):** GPT-3.5 decomposes a claim into atomic facts. GPT-4 then writes a sentence *pair* that supports each fact only when combined, and a document containing those sentences. Negatives remove one sentence of a pair and are kept only if GPT-4 judges the fact unsupported. Subclaims are built from all non-empty subsets of atomic facts.
- **D2C (doc→claim):** a human document is split into 3 chunks, GPT-4 summarises each, the summaries are decomposed into subclaims, and labelled negatives come from removing one sentence at a time. Subclaims are also checked against the *other* chunks. — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- MiniCheck training data is about 14K synthetic examples plus a 21K "hard" ANLI subset (examples entailment models got wrong), 35K total; more ANLI did not help. Ablation on MiniCheck-FT5 (avg BAcc): full 74.7; without C2D 72.7; without D2C 71.5; **without both 59.9 (−14.8)**. — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- **CG2C (FactCG):** builds an entity-relation graph from a document, samples 3–4-hop sub-graphs, has an LLM write a claim covering them, then *removes a relation from the document* to create the negative. Labels come from the programmatic edit, not the LLM, so they are not capped by LLM labelling accuracy. Training is two-stage: C2D/ANLI + CG2C-MHQA, then D2C + CG2C-Doc. On the CoRe connected-reasoning test, FactCG-DBT scores 33.85 vs 26.54 for MiniCheck-DBT, but about half of its correct predictions still reflect "disconnected" (shortcut) reasoning. — [FactCG paper](https://arxiv.org/html/2501.17144)
- **Token-level span detection** (LettuceDetect) labels each answer token supported or hallucinated, then derives example-level labels from spans. It reaches 79.2 example F1 on RAGTruth and beats prompt-based GPT-4-turbo by about 16 F1. — [LettuceDetect paper](https://arxiv.org/html/2502.17125v1)
- **Error-type-targeted synthetic data** (TinyLettuce): the generator injects specific error types (numerical, temporal, …) at controlled intensity. A 17M model trained on 3K such samples hits 90.9 F1 on same-distribution synthetic data but 68.5 on RAGTruth. — [TINYLETTUCE.md](https://cdn.jsdelivr.net/gh/KRLabsOrg/LettuceDetect@main/docs/TINYLETTUCE.md)
- **Decomposition is double-edged.** "Decomposition Dilemmas" (arXiv 2411.02400) finds a trade-off between accuracy gains and decomposition noise, with mixed results across studies. DnDScore (arXiv 2412.13175) shows atomic facts can lose the entity context needed for verification. — [Decomposition Dilemmas](https://arxiv.org/pdf/2411.02400v1); [DnDScore](https://export.arxiv.org/pdf/2412.13175)
- MiniCheck found GPT-4 decomposition or decontextualisation raised GPT-4 cost by 1.5–2× ($161–212 vs $107), and its own models work at sentence level without LLM decomposition. — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- **Numbers:** NLI models largely fail quantitative entailment (EQUATE). Symbolic quantity handling helps numbers but hurts verbal reasoning when used alone, which points to a hybrid. — [EQUATE](https://aclanthology.org/K19-1033/)
- ORION combines retrieval over long context with a 512-token NLI encoder and reports RAGTruth response-level F1 0.83. That suggests "retrieve the relevant KB chunk, then entail" lets a short-window encoder compete. — [ORION](https://arxiv.org/abs/2504.15771)
- Luna uses sliding-window chunking so its 512-token DeBERTa handles long contexts. — [Luna](https://arxiv.org/abs/2406.00975v1)

### Inferences (for Decide)
- **Pipeline sketch that fits 25 ms:**
  1. Split the draft into sentences or clauses; support replies are usually 1–6 sentences.
  2. Split the KB into roughly 128–256-token chunks.
  3. Prefilter with lexical/BM25 or embedding overlap to the top 1–2 chunks per draft sentence.
  4. Run the cross-encoder on (chunk, sentence) pairs.
  5. Score each sentence as the max over chunks; score the draft as the min over sentences, with "share supported" as a soft feature.
  6. Run the symbolic number/duration/scope checks per sentence and OR them in as hard vetoes.

  This keeps the number of encoder calls small (about 2–10 short pairs). With an fp32 17M model that is plausible within budget, but I have no sourced measurement.
- **Hard-negative generation for our domain:** apply the C2D/D2C/CG2C idea to KB articles. Have an LLM write faithful multi-sentence support replies from a KB article. Create negatives by (i) deleting the KB sentence that supports one reply sentence, (ii) perturbing one slot (number, unit, audience, scope, frequency, plan tier, negation), and (iii) adding plausible "baseless info" (extra promises, policies). Include positives that paraphrase heavily ("two weeks" vs "14 days") so the model doesn't learn "lexical difference = unsupported". This directly targets the one-slot shortcut our hackathon data may have taught.
- **Partial support:** sentence-level max/min aggregation handles multi-claim drafts naturally. Token-level span labelling (LettuceDetect-style) would also explain *which* phrase is unsupported, which helps calibration and debugging.
- **Non-claims:** greetings, empathy and sign-offs ("Sorry for the trouble!") should not be checked. Exclude them by a simple classifier or rule, or treat them as supported. Otherwise min-aggregation produces false "ungrounded" flags. This is an inference; I found no source specifically on support-reply boilerplate.

### Gaps
- MiniCheck's exact inference chunk size (words/tokens) was not in the fetched text.
- No source measures sentence-split + max-over-chunks with a 17M encoder on CPU.
- No source evaluates scope/audience-broadening ("all plans" vs "all paid plans") specifically. It is closest to "subtle conflict" in RAGTruth.

---

## 4. Calibration, abstention, threshold transfer across datasets

### Takeaway
A fixed 0.5 threshold on modern checkers costs only about 0.5–2 BAcc points on average compared with per-dataset tuned thresholds (MiniCheck-FT5 75.0 vs 75.5; FactCG-DBT 75.6 vs 77.2; AlignScore 70.5 vs 72.5). Per-*dataset* optimal thresholds still vary widely, and older detectors degrade significantly without in-domain tuning. Calibration (ECE) is rarely reported for these models.

### Cited Findings
- MiniCheck uses a fixed t=0.5 with no per-dataset tuning. The authors note most checkers output scores near the extremes, so small threshold changes matter little, and threshold-tuned results (App. B.1) are "qualitatively similar". — [MiniCheck paper](https://arxiv.org/html/2404.10774)
- FactCG tunes per-dataset thresholds to 2 decimals on validation data. It notes a unified threshold "may underestimate the model" and leaves unified-threshold calibration as future work.
  - Tuned vs 0.5: FactCG-DBT 77.2 vs 75.6; FactCG-FT5 76.7 vs 75.0; MiniCheck-FT5 75.5 vs 75.0; MiniCheck-DBT 74.1 vs 73.1; AlignScore 72.5 vs 70.5. — [FactCG paper](https://arxiv.org/html/2501.17144)
- DEEP (arXiv 2406.13009): prior factual-error detectors "perform significantly worse without optimizing the thresholds on subsets of the evaluated dataset". It proposes calibrated ensembles of LLM prompts "to produce empirically accurate probabilities", avoiding "thresholding techniques not available in practical settings". The abstract gives no numbers. — [DEEP](https://arxiv.org/abs/2406.13009)
- "Understanding Factual Errors in Summarization" (arXiv 2205.12854) compares threshold-per-dataset vs single-threshold settings with BAcc, because consistent/inconsistent classes are imbalanced. — [Tang et al. 2022](https://arxiv.org/pdf/2205.12854)
- The HHEM model card gives scores in [0,1] but no calibration claims. It notes order asymmetry: premise and hypothesis must not be swapped. — [HHEM card](https://huggingface.co/vectara/hallucination_evaluation_model)

### Inferences (for Decide)
- Decide's 100% dev accuracy on one-slot perturbations says little about threshold transfer. Calibrate on a *held-out, differently generated* set: synthetic KB-reply pairs from a different LLM or prompt style, or RAGTruth-QA. Use temperature scaling or isotonic regression on the final fused score, not just the cross-encoder logit.
- Use a **three-way output: supported / unsupported / abstain (low confidence)**. Abstain when the cross-encoder score is mid-range and no symbolic veto fired, or when KB coverage is low (no chunk passes the retrieval prefilter). Abstain bands chosen on held-out data transfer better than a single cut-point, though no source verifies this.
- int8 broke number precision for us. Keep the encoder fp32 (or fp16/bf16 where supported) and let the symbolic number checker handle exact values. The learned model then only needs "does this sentence's *proposition* match", not digit-exact comparison. EQUATE suggests encoders are weak at the latter anyway.

### Gaps
- I found no ECE or reliability-diagram numbers for MiniCheck, FactCG, AlignScore, HHEM or LettuceDetect.
- I did not retrieve DEEP's quantitative threshold-transfer results (abstract only).

---

## 5. Can we distil or fine-tune our 17M encoder on public NLI/grounding data, and what licences apply?

### Takeaway
Yes, and there is direct precedent: TinyLettuce fine-tuned Ettin-17M on RAGTruth plus 3K synthetic samples (MIT, 68.5 RAGTruth F1). The commercially clean data is MiniCheck C2D/D2C (MIT, 14.4K), RAGTruth (MIT), and our own LLM-generated KB/reply pairs. ANLI is CC BY-NC 4.0 (non-commercial) and needs care. The training-data ablations show synthetic hard negatives matter far more than more NLI.

### Cited Findings
- **MiniCheck C2D/D2C dataset** (`lytang/C2D-and-D2C-MiniCheck`): MIT, 14,395 rows (c2d 7.08K, d2c 7.32K), fields claim/doc/label. The card does not name the generator; the paper says GPT-3.5/GPT-4 generated it. — [HF dataset](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck); [MiniCheck paper](https://arxiv.org/html/2404.10774)
- **ANLI:** "licensed under Creative Commons-Non Commercial 4.0". — [ANLI GitHub](https://github.com/facebookresearch/anli)
- **RAGTruth:** MIT licence, about 18K labelled responses with span annotations. — [RAGTruth GitHub](https://github.com/ParticleMedia/RAGTruth)
- **FactCG CG2C data** draws on HotpotQA and MuSiQue (CC BY 4.0) and MiniCheck D2C docs. The paper lists MiniCheck as Apache-2.0 and AlignScore as MIT. — [FactCG paper](https://arxiv.org/html/2501.17144)
- **TinyLettuce recipe:** Ettin-17M, RAGTruth plus 3K synthetic (GPT-OSS-120B generator, error types such as numerical/temporal), lr 1e-5, 5 epochs, bs 16, max length 4096. Results: 68.5 RAGTruth F1 and 90.9 synthetic F1. — [TINYLETTUCE.md](https://cdn.jsdelivr.net/gh/KRLabsOrg/LettuceDetect@main/docs/TINYLETTUCE.md)
- **Teacher model licences:** HHEM-2.1-Open is Apache-2.0 ([card](https://huggingface.co/vectara/hallucination_evaluation_model)). Granite Guardian is Apache-2.0 ([IBM](https://www.ibm.com/granite/docs/models/guardian)). LettuceDetect and TinyLettuce are MIT ([README](https://cdn.jsdelivr.net/gh/KRLabsOrg/LettuceDetect@main/README.md)). Bespoke-MiniCheck-7B needs a commercial licence from Bespoke ([MiniCheck GitHub](https://github.com/Liyan06/MiniCheck)). Lynx-70B is reportedly CC-BY-NC-4.0 ([featherless](https://featherless.ai/models/PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct/readme)).
- MiniCheck found more ANLI beyond the 21K hard subset "not effective". Removing both synthetic sets cost 14.8 BAcc. — [MiniCheck paper](https://arxiv.org/html/2404.10774)

### Inferences (for Decide)
- **Suggested training mix for the shared Ettin-17M grounded head:**
  1. MiniCheck C2D/D2C (MIT).
  2. RAGTruth-QA and summarisation pairs converted to (context, response sentence) labels (MIT).
  3. Our own synthetic KB→reply set with hard negatives (deleted-evidence, single-slot perturbations, scope broadening, added baseless promises) plus paraphrase positives.
  4. Optionally soft labels from an Apache/MIT teacher (HHEM-2.1-Open, FactCG-DBT or MiniCheck-RBTA/DBTA, checking each licence) for distillation.

  Avoid ANLI if the project could go commercial; it is fine for a research hackathon under CC BY-NC.
- **Multi-task interference risk:** because Decide's encoder is shared across decisions, consider a small task-specific adapter or head and/or replay of other tasks' data, so grounding fine-tuning doesn't degrade other heads. This is general multi-task practice, not sourced here.
- **Validation for "different dataset" robustness:** hold out RAGTruth-QA, LLM-AggreFact ClaimVerify/LFQA, and a synthetic KB set generated by a *different* LLM. Report BAcc at a fixed threshold, plus ECE. Expect a 17M model to land around TinyLettuce-17M levels (~68 F1 on RAGTruth) unless domain-specific synthetic data closes the gap. The 68M Ettin variant (~75 F1) is the obvious upgrade if latency allows.

### Gaps
- MNLI and DocNLI licences were not verified in this session.
- No public result was found for distilling a 0.4B checker (MiniCheck/FactCG) into a ≤20M encoder; the size–accuracy curve is only known from TinyLettuce (17M/32M/68M).
- Whether GPT-4-generated data (MiniCheck) carries OpenAI terms-of-use restrictions on training competing models is unclear; the dataset card says MIT only.
