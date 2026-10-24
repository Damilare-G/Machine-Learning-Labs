# Lab 3: Architectures and their costs

## 1. Goal
Compare three architectures of similar scale — MLP, CNN, and patch‑Attention — on Fashion‑MNIST, evaluating their parameter count, FLOPs, latency, and accuracy, and explaining the trade‑offs using Chapter 4 access‑pattern concepts.

## 2. Method
- **Data:** Fashion‑MNIST, Normalize((0.2860,), (0.3530,)), 55k train / 5k val (seeded split) / official 10k test, batch 128.
- **Models:**  
  - MLP  
  - CNN  
  - Patch‑Attention (learned positional embedding added; Transformers are permutation‑invariant, so positional encoding is required to preserve spatial meaning).
- **Cost:**  
  - torchinfo params  
  - hand FLOPs (photos of paper calculations: `results/hand_calc_*.jpg`)  
  - thop FLOPs (2×MACs)  
  - latency = median of 100 runs (batch 1) and 20 runs (batch 128) after 1 warm‑up
- **Training:** 5 epochs, Adam lr 1e‑3, batch 128, seed 42.
- **Machine / CPU threads / device:** CPU, 4 threads.

## 3. Results

| Model | Params | Hand FLOPs | thop FLOPs (2xMACs) | Latency b1 (ms) | Latency b128 (ms) | Test acc |
|---|---|---|---|---|---|---|
| **MLP** | 235,146 | 469,504 | 469,504 | 0.0913 | 0.7513 | 0.8780 |
| **CNN** | 20,490 | 2,063,488 | 2,063,488 | 0.3678 | 17.4426 | 0.8961 |
| **Attention** | 71,818 | 2,329,856 | 101,632 | 1.0045 | 16.5194 | 0.8546 |

**Plot:** `results/tradeoff.png`

### Hand‑vs‑thop gap
- **MLP** and **CNN** match perfectly because their operations (dense matmul and convolution) are fully supported by thop.
- **Attention** shows a **22.92× gap** because thop does **not** count:
  - QKᵀ multiplication  
  - softmax  
  - multi‑head splitting  
  - V projection  
  - attention score computation  
  - feed‑forward layers inside TransformerEncoder  
- Therefore, thop severely underestimates true FLOPs for attention.

### Parameter target note
None of the models fall in the **100k–150k** target range:
- MLP is **too large** (235k)  
- CNN is **too small** (20k)  
- Attention is **below target** (72k)  

This is acceptable as long as the architectures remain comparable and analysis is correct.

## 4. Three conclusions (own words)
1. **CNN is the most cost‑effective architecture**: it achieves the highest accuracy with the fewest parameters due to spatial weight sharing and locality.
2. **MLP is the fastest model** despite having the most parameters, because dense matrix multiplications have excellent CPU locality and low FLOPs.
3. **Attention is the least efficient for small images**: it has moderate parameters but very high true FLOPs and poor locality, leading to the slowest latency and lowest accuracy.