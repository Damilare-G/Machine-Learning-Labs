# Individual Project Proposal: On-Device Hausa Keyword Spotting with INT8 Quantization

**Student:** Oyefara Damilare  **Student ID:** 73599

## Problem
Voice interfaces need a small, always-on model that detects a few spoken keywords on cheap hardware
without a cloud connection. Almost all keyword-spotting (KWS) datasets and models are in English, so
tens of millions of Hausa speakers in Nigeria and West Africa are left out. I will build a Hausa
keyword-spotting classifier that fits an edge-device budget while keeping close to the accuracy of a
full-precision model.

## Dataset
**Mozilla Common Voice, Hausa (scripted speech, latest release):** about 10k crowdsourced read-sentence
clips (about 12 hours) from about 60 speakers. License: CC0.
- **Building the keyword set:** no Hausa single-word dataset exists, so I will create one:
  1. Count word frequencies in the transcripts and pick the **10 most frequent content words**
     (at least two syllables, not short function words), each with ≥ 150 occurrences.
  2. Find each word's start and end time with **forced alignment** (torchaudio MMS aligner, which supports Hausa).
  3. Cut a 1-second window centred on each word.
  4. Add two extra classes, as in Speech Commands: **"unknown"** (other words) and **"silence"** (background noise).
     That gives 12 classes.
- **Splits:** the official Common Voice train/dev/test splits, which have no speakers in common, so the
  test results reflect unseen voices.
- **Features:** 40-band log-Mel spectrogram (about 49 × 40). Training augmentation: time shift and added noise.
- **Licence note:** Common Voice must not be re-hosted, so the repo will contain only scripts and the
  list of keywords, not the audio.

## Success Metrics
| Metric | Target |
|---|---|
| Top-1 accuracy on the test split (12 classes) | ≥ 85%, and ≤ 1 pp drop vs. FP32 baseline |
| Latency (batch 1, single CPU thread, features + inference) | p95 ≤ 50 ms |
| Model size on disk | ≤ 5 MB |

Latency: 100 warm-up runs, then 1,000 timed runs; I will report p50 and p95.
Size: the size of the saved TorchScript file. I will also report per-class accuracy and a confusion matrix.

## Deployment Constraints
- Target: CPU-only edge device (laptop CPU limited to 1 thread, as a stand-in for a Raspberry Pi-class board).
- **Latency ≤ 50 ms** per one-second window, so the model can run in real time on streaming audio.
- **Size ≤ 5 MB**, so it fits in flash memory alongside the application.

## Planned Optimization: Quantization
1. **Baseline:** train a ResNet-style CNN (about 2M parameters, about 8 MB in FP32). This is over the size budget.
2. **Post-training static quantization (PTQ):** quantize weights and activations to INT8 with PyTorch
   (`torch.ao.quantization`), calibrating on about 500 training clips. Expected result: about 4× smaller (about 2 MB) and faster.
3. **Quantization-aware training (QAT):** use this only if PTQ loses more than 1 pp of accuracy.
4. **Comparison table:** FP32 vs. PTQ vs. QAT on accuracy, p50/p95 latency and size.

## Risks
- **Too few examples per keyword:** if fewer than 150 per word, use fewer keywords (e.g., 8) or add
  Hausa speech from another open corpus.
- **Bad alignments:** listen to a random sample of 50 cut clips and drop words that are aligned badly.
- **Unsupported INT8 layers:** keep those layers in FP32.
