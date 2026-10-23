# Small Fast Text Encoders, CPU Inference, and Multi-Task Shared-Encoder Learning (for "Decide")

Scope: state of the art for 17M–400M encoders (2024–Oct 2026), CPU inference tricks (quantisation, runtimes), how to spend ~20 ms of latency headroom, multi-task optimisation on a shared encoder, and cross-encoder vs bi-encoder vs late interaction for (draft, KB) grounding. Current as of 2026-10-24. Where only vendor or author numbers exist, these notes say so.

---

## 1. Which encoders (2024–2026) are best at 17–150M for classification and cross-encoding?

### Takeaway
Ettin (July 2025) is still the best open family in the 17–150M range with matched sizes. On GLUE it climbs steeply with size: 17M 79.2 → 32M 83.5 → 68M 87.2 → 150M 88.9. That gap is large on clean benchmarks, yet Decide saw flat quality across sizes. So our ceiling is label noise or the task, not model capacity. Two newer releases are worth a look at 150M+: UltraBERT-150M (Oct 2026), which beats Ettin-150M on its authors' own benchmarks, and LFM2.5-Encoder-230M (Jul 2026, CPU-oriented). For the grounded pair task, the Ettin cross-encoder rerankers (May 2026) are a strong ready-made initialisation at 17M and 32M.

### Cited Findings
**Ettin (Seq vs Seq), arXiv 2507.11412**
- Paired encoder and decoder models from 17M to 1B, trained on up to 2T tokens with the same recipe. The encoders beat ModernBERT. A 400M encoder beats a 1B decoder on MNLI, and continued training to switch objective (decoder→encoder) is "subpar" — [arXiv abstract](https://arxiv.org/abs/2507.11412)
- GLUE avg / MNLI: Ettin-17M 79.2/79.5; 32M 83.5/83.4; 68M 87.2/87.0; 150M 88.9/89.2; 400M 90.8/91.3; 1B 91.6/91.8. Baselines: ModernBERT-base 88.4/89.1, ModernBERT-large 90.4/90.8, MiniLM-L12 84.6/85.6, BERT-base 84.7/85.4, DistilBERT 81.5/82.7, TinyBERT 77.0/80.9, BERT-small 79.0/79.2 — [Ettin paper HTML, Table 7](https://arxiv.org/html/2507.11412v2)
- MTEB v2 retrieval: Ettin 17M 35.6, 32M 39.7, 68M 43.1, 150M 45.7, 400M 48.4, 1B 50.1. Baselines: ModernBERT-base 43.9, MiniLM-L12 38.4, BERT-base 41.2 — [Ettin paper HTML](https://arxiv.org/html/2507.11412v2)
- Architecture: 17M = 7 layers, hidden 256, FFN 384, 4 heads. 32M = 10 layers, hidden 384, FFN 576, 6 heads. Training is 1.7T pretrain + 250B mid-train + 50B decay — [Ettin paper HTML, Table 1](https://arxiv.org/html/2507.11412v2)
- Takeaway for Decide: Ettin-68M is about MiniLM-L12 size with a GLUE avg of 87.2, near ModernBERT-base (88.4) — [Ettin paper HTML](https://arxiv.org/html/2507.11412v2)

**Ettin reranker family (cross-encoders), May 19 2026**
- Six Sentence-Transformers CrossEncoders (17M–1B), Apache-2.0, built by pointwise MSE distillation of raw logits from mxbai-rerank-large-v2 (1.54B) on about 143M (query, doc, score) triples. One epoch. The training script is about 150 lines — [HF blog](https://huggingface.co/blog/ettin-reranker)
- NanoBEIR NDCG@10: Ettin-17M 0.6746, 32M 0.6825, 68M 0.6915, 150M 0.7086, 400M 0.7193, 1B 0.7237. For comparison: bge-reranker-v2-m3 (568M) 0.6971, ms-marco-MiniLM-L6-v2 (22.7M) 0.6312, MiniLM-L12 0.6369, bge-reranker-large 0.6379 — [HF blog](https://huggingface.co/blog/ettin-reranker)
- CPU throughput on an i7-13700K: Ettin-17M 267.4 pairs/s, 1B 2.1 pairs/s — [HF blog](https://huggingface.co/blog/ettin-reranker)

**ModernBERT, arXiv 2412.13663 (Dec 2024)**
- 2T tokens, native 8192 context. Claimed as a "major Pareto improvement over older encoders" and "the most speed and memory efficient encoder", but "designed for inference on common GPUs" — [arXiv abstract](https://arxiv.org/abs/2412.13663)
- Its unpadding and Flash-Attention speedups are inherited by mmBERT ("more than 2x faster… on variable-length sequences") — [mmBERT paper](https://arxiv.org/html/2509.06888v1)

**UltraBERT (dmis-lab), Oct 14 2026, Apache-2.0**
- Sizes 150M and 380M, plus an Embed-150M variant, with native 32K context. Author-reported results at base size (UltraBERT-150M vs Ettin-149M vs ModernBERT-149M): GLUE 89.5 / 88.9 / 88.4; BEIR 45.2 / 42.4 / 42.6; RewardBench 73.0 / 70.3 / 68.0. At medium size: GLUE 91.0 vs Ettin-395M 90.8 vs NeoBERT-222M 89.0. No separate NLI numbers. No independent replication yet. No size below 150M — [HF blog](https://huggingface.co/blog/dmis-lab/ultrabert)

**LFM2.5-Encoders (Liquid AI), Jul 28 2026**
- 230M and 350M encoders, converted from LFM2.5 decoders (bidirectional mask, non-causal short convolutions, 30% MLM), with 8K context. Vendor claims: the 230M beats ModernBERT-base and every EuroBERT model on 17 fine-tuned GLUE/SuperGLUE/multilingual tasks (mean of 5 seeds), and is the fastest tested model on CPU at short sequence lengths. At 8K tokens it takes about 28 s per pass on CPU vs over 90 s for ModernBERT-base. Numbers appear only in charts, and Ettin is not compared — [HF blog](https://huggingface.co/blog/LiquidAI/lfm2-5-encoders), [Liquid blog](https://www.liquid.ai/blog/lfm2-5-encoders)
- Caveat: Ettin found that decoder→encoder conversion is subpar compared with native encoder training ([Ettin](https://arxiv.org/abs/2507.11412)). Treat the LFM2.5 claims as unverified until there is a head-to-head against Ettin.

**NeoBERT, arXiv 2502.19587 (ICLR 2026)**
- 250M parameters, 4096 context, MIT license. Claims to beat BERT-large, RoBERTa-large, NomicBERT and ModernBERT under identical fine-tuning, with an "optimal depth-to-width ratio" — [arXiv](https://arxiv.org/abs/2502.19587v2), [HF](https://huggingface.co/chandar-lab/NeoBERT)
- The UltraBERT comparison reports NeoBERT-222M at 89.0 GLUE, below Ettin-395M (90.8) and ModernBERT-large (90.4) — [UltraBERT blog](https://huggingface.co/blog/dmis-lab/ultrabert). Not a fit for Decide's size range.

**mmBERT (multilingual), arXiv 2509.06888**
- small = 140M total / 42M non-embedding. base = 307M / 110M non-embedding. 256K Gemma-2 vocabulary. English GLUE: small 84.7, base 86.3, vs ModernBERT-base 87.4 and EuroBERT-210m 81.2 — [mmBERT paper](https://arxiv.org/html/2509.06888v1)
- Relevance: only if judges use non-English tickets. Most of mmBERT-small's parameters are embedding table, which costs file size but little compute.

**Static embeddings (Model2Vec / POTION)**
- potion-base-8M is distilled from bge-base-en-v1.5. The newer results page gives MTEB avg 51.08 and classification 70.34, vs all-MiniLM-L6-v2 at 55.93 and 69.25 — [Model2Vec results](https://minish.ai/packages/model2vec/results/). An older blog gives different numbers (50.03 and 64.44 vs 56.09 and 62.62) — [POTION blog](https://minish.ai/blog/2024-10-29-tokenlearn-blogpost/). The tables disagree, probably because of MTEB versions.
- Static embeddings have no token interaction, so they cannot do cross-encoding (draft vs KB entailment). They fit only as a sub-millisecond pre-filter or fallback — [potion-base-8M card](https://huggingface.co/minishlab/potion-base-8M)

**Other retrieval-oriented small encoders**
- IBM Granite Embedding R2 uses a ModernBERT-style architecture. Its multilingual small variant has 97M parameters — [Granite R2 paper](https://arxiv.org/pdf/2508.21085)

### Inferences
- On clean GLUE, Ettin gains +4.3 points from 17M to 32M and +3.7 from 32M to 68M. Decide saw no gain from 17M to 150M. That strongly implies our metric is capped by label noise or task ambiguity, so model size is not the lever on our own dataset. On an unseen judge dataset with more varied language, though, the larger models' better transfer (the GLUE/MNLI gap) may show up. One robustness hedge is to ship Ettin-32M or 68M if latency allows.
- For grounded, start from the Ettin-17M or 32M reranker checkpoint instead of the raw MLM checkpoint. It has already learned query–document relevance from 143M distilled pairs. Relevance is not entailment, so it still needs fine-tuning on our labels.
- Ettin-17M is 7 layers at hidden 256, so CPU cost is tiny. A 68M model (roughly 4× the compute) should still fit well under 25 ms p95 at short sequence lengths. That is an extrapolation from our 2.5–5.4 ms figure, not a measured number.

### Gaps
- No independent head-to-head of UltraBERT, LFM2.5-Encoder and Ettin at 150M or below. UltraBERT has no sub-150M model.
- No GTE-modernbert, Arctic-embed-s/xs, jina-v3 small or BGE-small numbers were fetched in this pass. They are retrieval bi-encoders and less relevant to fine-tuned classification.
- No new sub-50M encoder released in 2026 that beats Ettin-17M/32M was found.

---

## 2. How to spend the latency headroom: ensembles, multi-sample dropout, TTA, early exit

### Takeaway
The evidence says a single larger model usually matches an ensemble's calibration and robustness benefits. Committees of small models can beat one bigger model at equal compute, but that evidence comes mostly from vision. For Decide, the best-supported use of ~20 ms is one of two options. (a) Move up to Ettin-32M or 68M. (b) Run a 2–3 member ensemble of differently seeded or differently sized 17M/32M models, then apply temperature scaling to the averaged output. Multi-sample dropout is a training-time trick with no inference cost. Early exit saves latency, which we don't need, and is not a calibration tool.

### Cited Findings
- **Committees vs bigger models**: two EfficientNet-B5 models averaged match B7 accuracy at about half the FLOPs (20.5B vs 37B). Cascades are 5.4× faster than B7 at equal accuracy, and the speedups carry over to real on-device latency. The paper covers vision and video, not NLP — [Wang et al., Wisdom of Committees, arXiv 2012.01988](https://arxiv.org/pdf/2012.01988); [Google Research blog](https://research.google/blog/model-ensembles-are-faster-than-you-think/)
- **Ensembles may be unnecessary**: "a single (but larger) neural network can replicate" the uncertainty and dataset-shift robustness credited to deep ensembles. Ensemble diversity "does not meaningfully contribute" to OOD detection, and ensemble OOD gains are "strongly determined by their in-distribution performance" — [Abe et al., NeurIPS 2022, arXiv 2202.06985](https://arxiv.org/abs/2202.06985v1)
- **Calibrating ensembles**: temperature scaling works best when applied after averaging the members, and roughly halves ECE in low-data settings. Pooling without calibration can make ensembles under-confident. These are vision experiments — [Uncertainty Quantification and Deep Ensembles, NeurIPS 2021](https://nips.cc/virtual/2021/poster/27586)
- **Cheap ensemble on a shared backbone**: one shared fine-tuned backbone with several differently seeded classifier heads — [Torch-Uncertainty, arXiv 2511.10282](https://arxiv.org/pdf/2511.10282)
- **Model soups** (weight averaging) hurt grounded in our own runs. Output-averaging ensembles do not need the models to share a loss basin, so they avoid that failure mode. This is an inference.
- **Multi-sample dropout**: draws several dropout masks after the encoder, computes and averages the loss over shared-weight heads, and converges in fewer iterations with lower train and validation error. The extra compute is small because only the post-dropout layers are duplicated. At inference dropout is off, so the cost is zero — [Inoue, arXiv 1905.09788](https://arxiv.org/pdf/1905.09788)
- **Early exit**: DeeBERT exits when the entropy of an off-ramp falls below a threshold. Ensembling several layers into one off-ramp gave "no significant gain" because errors are shared across layers — [DeeBERT, arXiv 2004.12993](https://arxiv.org/pdf/2004.12993). PABEE exits after k consecutive internal classifiers agree. A survey credits it with some adversarial robustness from an ensemble effect — [Early-exit survey, arXiv 2501.07670](https://arxiv.org/pdf/2501.07670). Exit signals depend on calibrated internal classifiers, and recent work treats calibration as something early exit needs, not something it delivers — [COSEE, arXiv 2412.13236](https://arxiv.org/pdf/2412.13236); [Null-space projection certainty, arXiv 2506.17249](https://arxiv.org/pdf/2506.17249); [BEEM, arXiv 2502.00745](https://arxiv.org/pdf/2502.00745)

### Inferences
- Recommended ladder, to measure on our own CPU:
  1. Ettin-32M or 68M as a single model, as the robustness hedge for an unseen dataset (supported by Abe et al.).
  2. If staying small, average a 3-seed Ettin-17M ensemble at the logit level, which is about 3× the current cost (roughly 8–16 ms), then temperature-scale it per head. This also gives a disagreement signal for an "abstain or escalate" story. For approve's 3:1 cost, set the threshold on the calibrated probability (threshold = 1/(1+3) = 0.25 on P(costly class), or the equivalent for our cost direction).
  3. Train with multi-sample dropout (5–8 masks) for free regularisation.
- Skip early exit: we are already about 20 ms under budget, and early exit trades accuracy for speed. PABEE-style layer agreement could still serve as an extra confidence feature.
- TTA for text (paraphrase or truncation variants) has little primary evidence for encoder classifiers, and it is risky for grounded because it could change numbers.

### Gaps
- No primary NLP study found that compares an ensemble of small fine-tuned encoders with one larger encoder on ECE at equal CPU latency. Our own A/B test would be the evidence.
- No good primary source for test-time augmentation with encoder text classifiers.

---

## 3. CPU inference tricks: ONNX Runtime, quantisation done right, OpenVINO, caching

### Takeaway
Dynamic int8 hurt grounded (99%→92%). That fits the known transformer failure mode: a few hidden dimensions carry huge activation outliers, especially the residual sum after the FFN. Per-tensor 8-bit scales then wipe out the fine distinctions, such as exact numbers, that grounding depends on. Fixes, in order of effort: per-channel weights with S8S8 QDQ; excluding the FFN-output, residual and LayerNorm, embedding and classifier nodes; ORT's qdq_loss_debug to find the worst tensors; QAT or outlier-suppressing pretraining. With 20 ms of headroom, fp32 is the right default, and int8 is only worth it to fit a bigger model.

### Cited Findings
- **ORT guidance**: dynamic quantisation is recommended for transformers and static for CNNs. Run the transformer optimiser first (fusions such as QAttention). Per-channel "can improve accuracy when weight ranges are large". On AVX2/AVX512 machines without VNNI, U8S8 can saturate (VPMADDUBSW), so use reduce_range (7-bit weights) or U8U8. S8S8 QDQ is the recommended default. Calibration options are MinMax, Entropy and Percentile. `qdq_loss_debug` (`create_activation_matching` etc.) compares fp32 and int8 activations tensor by tensor to find the worst layers. Older hardware can be slower with int8 because of quantise/dequantise overhead — [ORT quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- **Excluding nodes**: `nodes_to_exclude` exists in ORT's quantize_dynamic/quantize_static API and StaticQuantConfig — [ORT quantize.py source mirror](https://gemfury.com/ekyc/python:onnxruntime-gpu/-/content/quantization/quantize.py); [AMD Quark config reference](https://quark.docs.amd.com/release-0.8.2/onnx/appendix_full_quant_config_features.html)
- **Why transformers break under int8**: activations have very wide dynamic ranges, and structured outliers in the residual connections (linked to attention on [SEP]) are the main problem. A leave-one-out analysis singles out the residual sum after the FFN. Fixes are mixed precision, per-embedding-group quantisation, and QAT, with state-of-the-art BERT PTQ on GLUE — [Bondarenko et al., EMNLP 2021, arXiv 2109.12948](https://arxiv.org/pdf/2109.12948); [ACL Anthology](https://aclanthology.org/2021.emnlp-main.627)
- **Root cause and pretraining fix**: outliers come from attention heads learning a "no-op", which pushes softmax inputs ever larger. Clipped softmax or gated attention during pretraining removes them and allows full int8 activations "without any additional effort" — [Bondarenko et al., NeurIPS 2023, arXiv 2306.12929](https://arxiv.org/abs/2306.12929v2). This requires re-pretraining, so it doesn't apply to an Ettin checkpoint.
- **OpenVINO/NNCF**: NNCF 8-bit PTQ for BERT uses calibration data through `nncf.quantize` — [OpenVINO NNCF BERT tutorial](https://docs.openvino.ai/2023.3/notebooks/105-language-quantize-bert-with-output.html). Optimum-Intel plus NNCF int8 BERT gave about 2.95× over fp32 on Ice Lake "without much compromise in the accuracy" — [Microsoft Open Source blog 2023](https://opensource.microsoft.com/blog/2023/01/25/improve-bert-inference-speed-by-combining-the-power-of-optimum-openvino-onnx-runtime-and-azure/)
- One search result claimed that on AMX CPUs, OpenVINO fp16 weights with a bf16 runtime beat int8 up to batch 256. The arXiv ID it was attributed to (2606.09672) has an unrelated title, so this is **unverified and should not be cited** — [search result](https://arxiv.org/pdf/2606.09672)
- **Unpadding**: ModernBERT-family speedups come from unpadding and Flash-Attention on variable-length inputs — [mmBERT paper](https://arxiv.org/html/2509.06888v1). At batch size 1 on CPU there is no padding, so unpadding gives nothing. This is an inference.

### Inferences
- **Recommended quantisation recipe if int8 is still wanted** (for example, to afford a 68M model):
  1. Static QDQ, S8S8, per-channel weights.
  2. Calibrate with Percentile or Entropy on about 500 real tickets plus KB pairs that include numeric content.
  3. Exclude the embedding Gather, the final FFN output and residual Add, LayerNorm, Softmax, the classifier heads and the pooling.
  4. Run qdq_loss_debug and also exclude the top-k tensors by fp32/int8 error.
  5. If grounded still drops, run a short QAT (fake-quant) fine-tune, or keep the grounded path fp32 and quantise only route and approve. That second option needs a per-task graph.
- Weight-only int8 or int4 on MatMul (activations stay fp32) shrinks artifacts without touching activation precision, but gives little CPU speedup for small models.
- **KB caching**: a cross-encoder cannot cache KB encodings, because the draft and KB tokens attend to each other. Caching needs a bi-encoder or late-interaction model for grounded (see section 5). The ticket-only encoding for route and approve could be shared if those two use the same prompt, but they use typed prompts.
- **Thread settings**: on 4 cores at batch 1, set intra_op_num_threads to match physical cores, and turn off spinning if p95 jitter shows up. These are standard ORT settings, not sourced in this pass.

### Gaps
- No primary source fetched on ORT CPU fp16 or bf16 kernel coverage. Generally fp16 is not faster than fp32 on x86 without AMX or AVX512-FP16; this is unverified. bf16 wins need AMX (Sapphire Rapids and later) or Arm bf16 — check what hardware the judges use.
- No measured int8 vs fp32 accuracy results for numeric-sensitive NLI found. Token pruning (e.g. PoWER-BERT, LTP) was not researched in this pass. At short sequences with batch 1 it is unlikely to matter.

---

## 4. Multi-task learning on a shared encoder: loss balancing, prompts, adapters, adding new tasks

### Takeaway
Two independent NeurIPS 2022 studies found that well-tuned equal-weighted loss sums (unitary scalarization) match or beat PCGrad, GradNorm, CAGrad, Nash-MTL and similar methods once standard regularisation is applied. A NeurIPS 2023 follow-up confirmed that uniform scalarization is competitive at scale. For Decide's three tasks, plain weighted sums with tuned weights plus regularisation is the defensible choice. To add a new decision type cheaply, use a new head plus a per-task LoRA or adapter with the shared encoder frozen (zero interference), or rehearse old-task data while fine-tuning.

### Cited Findings
- **Kurin et al.**: unitary scalarization "coupled with standard regularization and stabilization techniques… matches or improves upon the performance of complex multi-task optimizers" in supervised and RL settings. Most of those optimizers need per-task gradients, adding memory, runtime and implementation overhead. The authors interpret the specialised optimizers partly as regularisers — [arXiv 2201.04122](https://arxiv.org/pdf/2201.04122); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4f301ae934f396086bfefd1139039dbd-Abstract.html)
- **Xin et al. (Google)**: in large-scale language and vision experiments, multi-task optimisation methods "do not yield any performance improvements beyond what is achievable via traditional optimization approaches" (weighted loss averages). The paper also flags common training pitfalls and evaluation challenges — [arXiv 2209.11379](https://arxiv.org/pdf/2209.11379); [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract.html)
- **Royer et al. (Qualcomm)**: uniform scalarization performs "on par" with costlier state-of-the-art MTO methods. They analyse why across task mixes and model sizes, and propose population-based training to search scalarization weights when there are many tasks — [arXiv 2310.08910](https://arxiv.org/pdf/2310.08910); [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html)
- Methods those studies tested against (primary papers, for citation): PCGrad [arXiv 2001.06782](https://arxiv.org/abs/2001.06782); GradNorm [arXiv 1711.02257](https://arxiv.org/abs/1711.02257); uncertainty weighting (Kendall et al.) [arXiv 1705.07115](https://arxiv.org/abs/1705.07115); CAGrad [arXiv 2110.14048](https://arxiv.org/abs/2110.14048); Nash-MTL [arXiv 2202.01017](https://arxiv.org/abs/2202.01017). These links were not fetched in this pass; the IDs are from prior knowledge.
- **Multi-task LoRA**: plain shared LoRA projects all tasks into one low-rank space and blurs task-specific information. MTL-LoRA adds a learnable per-task transformation in that space — [arXiv 2410.09437](https://arxiv.org/pdf/2410.09437). LoRI freezes A as random projections and sparsifies B with task-specific masks to cut cross-task interference, and targets merging and continual learning — [arXiv 2504.07448](https://arxiv.org/pdf/2504.07448v2). Ortho-LoRA (gradient projection) reports recovering about 95% of the multi-task vs single-task gap on GLUE — [arXiv 2601.09684](https://arxiv.org/html/2601.09684v1)
- **Weight averaging**: in our own runs, soups hurt grounded. This fits task-specific features being destroyed by averaging, and argues for keeping task-specific parameters (adapters or heads) separate rather than merged. This is an inference.

### Inferences
- **Decide recipe**:
  - Loss = w_route·CE + w_approve·cost-weighted CE + w_grounded·BCE, with weights tuned by a small grid or PBT, plus label smoothing (route has about 10% label noise) and multi-sample dropout. Cite Kurin, Xin and Royer when explaining why we do not use PCGrad or Nash-MTL. This is a "startup-level" justification backed by evidence.
  - For approve's 3:1 cost, a decision-theoretic threshold on calibrated probabilities is cleaner than loss re-weighting, and it transfers to a new dataset without retraining.
  - Task prefixes ("route:", "approve:", "grounded:") already give the encoder task conditioning. Keep them.
- **Adding a new decision type without interference**, from cheapest to most expensive:
  1. Frozen encoder plus a new head on pooled features: zero interference, weakest quality.
  2. Frozen encoder plus a per-task LoRA (rank 8–16) and a new head: zero interference, because the old tasks run without the adapter. LoRA can be merged per task at export, or kept as a separate small ONNX graph.
  3. Full multi-task re-fine-tune with replay of old-task data: best quality, needs regression checks on the old tasks.
- Gradient-conflict methods are not worth the complexity for 3 tasks, given the evidence above.

### Gaps
- No 2024–2026 evaluation found that is specific to small encoders (under 100M) with typed prompts versus separate heads.
- Whether a per-task LoRA on a 17M model (hidden 256) gives meaningful capacity is untested. Rank budgets at such small widths have not been studied in the sources.

---

## 5. Cross-encoder vs bi-encoder vs late interaction for (draft, KB article) grounding

### Takeaway
For "is this draft supported by this KB article", a cross-encoder is the correct architecture. Grounding/NLI needs token-level comparison of numbers and entities, and every state-of-the-art grounding checker found (MiniCheck, LettuceDetect) is a cross-encoder style model. Late interaction (ColBERT) is a CPU-friendly middle ground for *retrieving* the right KB article and allows caching KB token vectors. It is designed for relevance, not entailment. Keep the cross-encoder for the grounded decision.

### Cited Findings
- **MiniCheck** (EMNLP 2024): small fact-checkers trained on GPT-4-synthesised hard examples reach GPT-4-level accuracy at about 400× lower cost on LLM-AggreFact, a unified grounding benchmark. The best model is MiniCheck-FT5 (770M). The benchmark, data-generation code and models are public, with a leaderboard at llm-aggrefact.github.io — [arXiv 2404.10774](https://arxiv.org/html/2404.10774v2); [ACL Anthology](https://preview.aclanthology.org/setup/2024.emnlp-main.499)
- **LettuceDetect** (2025): a ModernBERT token-classification hallucination detector over (context, question, answer) trained on RAGTruth. The large model gets 79.22% example-level F1, +14.8% over Luna (the previous best encoder-based model), and is about 30× smaller than the best LLM-based detectors. It runs 30–60 examples/s on a GPU — [arXiv 2502.17125](https://arxiv.org/html/2502.17125v1)
- **Ettin cross-encoders** beat older MiniLM cross-encoders of similar size by about 4–5 NanoBEIR points (17M 0.6746 vs MiniLM-L6 0.6312) — [HF blog](https://huggingface.co/blog/ettin-reranker)
- **answerai-colbert-small-v1** (33M, Apache-2.0, late interaction): BEIR avg 53.79 vs ColBERTv2.0 at 50.02. The card claims it "vastly outperforms cross-encoders its size" as a reranker — [HF model card](https://huggingface.co/answerdotai/answerai-colbert-small-v1); [README revision](https://huggingface.co/answerdotai/answerai-colbert-small-v1/commit/f48715ef53c0e6cd21cb705e84a5371ebd73083e). Those are relevance benchmarks, not entailment ones, and the card predates the Ettin rerankers.

### Inferences
- **Robustness for judge datasets**:
  - Evaluate grounded on public grounding data as an external check: LLM-AggreFact, RAGTruth, and NLI sets like MNLI/ANLI.
  - Augment training with MiniCheck-style synthetic hard negatives. Perturb numbers, dates, amounts and entities in supported drafts to create unsupported ones. Precise numeric contradictions are exactly what our int8 run broke.
  - Truncation policy matters for long KB articles. Use chunk-level cross-encoding with max-over-chunks aggregation (the MiniCheck approach) rather than truncating at 512 tokens.
- **Architecture option** if judges supply many KB articles per ticket: ColBERT-style late interaction or a bi-encoder retrieves the top-k articles with cached KB vectors, then the cross-encoder makes the grounded decision on the top-1 to top-3. Under 25 ms on CPU looks plausible for 17M models, extrapolating from 267 pairs/s for Ettin-17M on an i7 CPU ([HF blog](https://huggingface.co/blog/ettin-reranker)), but this has not been measured.
- Token-level grounding (LettuceDetect style) could add explainability: highlighting which draft span is unsupported. That is a differentiator at the "research-paper level" bar, at extra cost of a token-classification head on the same encoder.

### Gaps
- No direct comparison found of ColBERT vs cross-encoder on NLI or grounding (as opposed to relevance) at small scale.
- MiniCheck's smallest-model numbers (e.g. RoBERTa/DeBERTa variants) and the current LLM-AggreFact leaderboard entries below 100M parameters were not fetched.
