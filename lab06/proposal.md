# Individual Project Proposal: Lightweight Hausa Sentiment Analysis via Knowledge Distillation

**Student:** Oyefara Damilare  **Student ID:** 73599

## Problem
Hausa is spoken by tens of millions of people in Nigeria and West Africa, but most sentiment-analysis
tools only support English. The best Hausa models are large transformers (100M+ parameters,
hundreds of MB), which are too big and slow for phones or low-cost servers. I will build a Hausa
sentiment classifier small and fast enough for on-device use, by distilling a large transformer
into a compact student model.

## Dataset
**NaijaSenti, Hausa subset** (Muhammad et al., LREC 2022; github.com/hausanlp/NaijaSenti):
human-annotated Hausa tweets (around 30k per language in the full corpus), labelled
**positive / neutral / negative**.
- The labels are already there; no data collection or annotation is needed.
- I will use the **official train / dev / test splits** without changes, so results can be compared with published baselines.
- Preprocessing: lowercase, replace @users and URLs with placeholder tokens, normalise repeated characters.
  Many tweets mix Hausa and English, so the tokenizer must handle both.

## Success Metrics
| Metric | Target |
|---|---|
| Macro-F1 on the official test set (3 classes) | Student within 3 pp of the teacher |
| Distillation gain | Student + KD beats the same student trained on labels only |
| Latency (batch 1, single CPU thread, tokenisation + inference) | p95 ≤ 50 ms |
| Model size on disk (weights + vocabulary) | ≤ 5 MB |

I use macro-F1 because the classes are imbalanced. Latency: 100 warm-up runs, then 1,000 timed
runs; I will report p50 and p95.

## Deployment Constraints
- Target: CPU-only (laptop CPU limited to 1 thread, as a stand-in for a phone or small server).
- **Latency ≤ 50 ms** per tweet.
- **Size ≤ 5 MB**, so it can ship inside a mobile app.

## Planned Optimization: Knowledge Distillation
1. **Teacher:** fine-tune a pretrained African-language transformer (AfriBERTa or AfroXLMR) on the
   training split. It is accurate but far over budget (>100M parameters, >300 MB).
2. **Student:** a small model trained from scratch: a 16k-word vocabulary with 64-dimensional
   embeddings, then a 1-D CNN, then a 3-way output. About 1.1M parameters, about 4.5 MB in FP32.
3. **Distillation loss:** a weighted mix of the usual loss on true labels and a KL-divergence loss
   that makes the student copy the teacher's softened predictions, with α = 0.5 and temperature T = 4.
   I will also test T ∈ {2, 4, 8}. The teacher's predictions are computed once and saved, so training the student stays cheap.
4. **Comparison table:** teacher vs. student trained only on labels vs. student with distillation,
   on macro-F1, latency and size. If needed, INT8 dynamic quantization of the student as a bonus step.

## Risks
- **Hausa–English mixing / spelling variation:** use subword (BPE) tokens instead of whole words if too many words are unknown.
- **Student too far below the teacher:** use a bigger student (up to the 5 MB limit), or use a BiLSTM instead of the CNN.
- **Not enough GPU for the teacher:** use AfriBERTa-small, or fine-tune on free Kaggle/Colab GPUs.
