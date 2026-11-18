# Individual Project Proposal: On-Device Keyword Spotting with INT8 Quantization

**Student:** Damilare Oyefara  **Student ID:** 73599

## Problem
Voice assistants listen all the time for a small set of command words ("yes", "no", "stop", "go", ...).
This detection has to run on cheap, low-power hardware with no cloud connection, so the model must be
fast and small. I will build a keyword-spotting (KWS) classifier that meets an edge-device budget
while keeping almost all the accuracy of a full-precision model.

## Dataset
**Google Speech Commands v0.02:** about 105k one-second, 16 kHz audio clips of 35 words from
about 2,600 speakers.
- Task: the standard 12-class setup, which is 10 keywords (yes, no, up, down, left, right, on, off, stop, go)
  plus "silence" and "unknown".
- Splits: the official `validation_list.txt` / `testing_list.txt`. They are split by speaker, so the
  same voice never appears in both train and test.
- Input features: 40-band log-Mel spectrogram (about 49 × 40 per clip). Training augmentation:
  time shift and background noise.

## Success Metrics
| Metric | Target |
|---|---|
| Top-1 accuracy on the official test set | ≥ 92%, and ≤ 1 pp drop vs. FP32 baseline |
| Latency (batch 1, single CPU thread, features + inference) | p95 ≤ 50 ms |
| Model size on disk | ≤ 5 MB |

Latency: 100 warm-up runs, then 1,000 timed runs; I will report p50 and p95.
Size: the size of the saved model file (TorchScript).

## Deployment Constraints
- Target: CPU-only edge device (laptop CPU limited to 1 thread, as a stand-in for a Raspberry Pi-class board).
- **Latency ≤ 50 ms** per one-second clip, so the model can run in real time on a streaming window.
- **Size ≤ 5 MB**, so it fits in flash memory alongside the application.

## Planned Optimization: Quantization
1. **Baseline:** train a ResNet-style CNN (about 2M parameters, about 8 MB in FP32). This is over the size budget.
2. **Post-training static quantization (PTQ):** quantize weights and activations to INT8 with PyTorch
   (`torch.ao.quantization`), calibrating on about 500 training clips. Expected result: about 4× smaller (about 2 MB) and faster.
3. **Quantization-aware training (QAT):** use this only if PTQ loses more than 1 pp of accuracy. It fine-tunes with simulated INT8 for a few epochs.
4. **Comparison table:** FP32 vs. PTQ vs. QAT on accuracy, p50/p95 latency and size, plus per-class accuracy
   to check which keywords suffer most.

## Risks
- Some layers may not support INT8. Fallback: keep those layers in FP32.
- Feature extraction might take up most of the latency. Fallback: precompute the Mel filterbank and use fewer bands.
