# Lab 6: Training pipelines and project proposal

## 1. Goal
Turn the best Lab 3 Fashion-MNIST model into a reproducible, YAML-driven training pipeline with seeding, checkpoints and MLflow tracking. Then use it to measure where training time goes (profiler, `num_workers`), how batch size, LR schedule and optimizer trade speed, memory and accuracy, check that gradient accumulation and checkpoint-resume behave correctly, and write the individual project proposal.

## 2. Method
- Model: **CNN** from Lab 3 (`Conv(1→16)-ReLU-MaxPool → Conv(16→32)-ReLU-MaxPool → Linear(1568→10)`, **20,490 parameters**), chosen because it was the best Lab 3 model by test accuracy (`config.yaml`: `model: CNN`). `lab06/config.yaml` is the single config object (any key can be overridden with `--set key=value`); seed 42; per-epoch seeds (`seed + epoch` for RNG and the shuffling sampler) make resume exact.
- Data: Fashion-MNIST, 55k train / 5k validation (split with seed 42) / 10k test. Default training: Adam, lr 1e-3, batch 128, `num_workers` 0, 5 epochs, no scheduler.
- Pipeline: `src/trainer.py` + `src/train.py`; checkpoints (`last.pt`, `best.pt`) plus final `model.pt` per run in `results/runs/<run>/`; MLflow logs params (all config keys + effective batch), per-epoch metrics (train/val loss, val acc, lr, epoch time, samples/s, per-phase time), final metrics (test acc, best/final val acc, peak memory, throughput) and artifacts (`config_used.yaml`, `history.csv`, `summary.json`, `model.pt`).
- Profiling: `torch.profiler` over one full epoch (430 batches, batch 128, `num_workers` 0) with `record_function` labels for each phase (`src/profile_epoch.py`); wall-clock phase timers recorded alongside for comparison.
- Studies (`src/studies.py`): workers {0,1,2,4,8} (2 epochs, persistent workers, epoch-2 steady state); batch {32..512} (5 epochs, Adam 1e-3); StepLR (step 3, γ 0.5) / CosineAnnealingLR / ReduceLROnPlateau (factor 0.5, patience 1) and SGD+momentum (lr 0.05, m 0.9) vs Adam (lr 1e-3) (10 epochs); accumulation 64x4 vs 256; interrupt at epoch 3 and resume (6-epoch run).
- Machine / device / PyTorch build / CPU threads: Windows 11 (build 10.0.26200), CPU only (no CUDA), Python 3.11.8, PyTorch 2.3.1+cpu, torchvision 0.18.1, MLflow 2.14.1 (`results/versions.txt`). CPU thread count was not recorded by the scripts.

## 3. Results
**Profile** (`results/profile_breakdown.csv`): data loading **56.6%** / forward **20.1%** / backward **20.5%** / optimizer **2.8%** (profiler time; wall-clock shares agree within 0.1 pp). Data loading is the largest single phase with `num_workers=0`: `enumerate(DataLoader)` is the top op (14.6 s inclusive), and the many tiny per-sample ops (`aten::_to_copy`, `div`, `eq`, `any`, ~55k calls each) are the ToTensor/Normalize transforms running in the main process. The heaviest compute ops are `max_pool2d_with_indices` and `convolution_backward` (~2.4 s each). Profiler overhead: the profiled epoch took 25.9 s against ~19.6 s for a normal epoch (~30% slower), and without the profiler the data share is about 49% (forward ~23.5%, backward ~24.5%, optimizer ~2.9%, from `best_model` history). The ranking stays the same either way.

**Workers** (`results/study_workers.csv`, `.png`):

| num_workers | epoch-2 time (s) | train samples/s | loader-only samples/s | data-wait share | bottleneck |
|---|---|---|---|---|---|
| 0 | 24.44 | 2,251 | 5,288 | 47.6% | data loading |
| 1 | 13.09 | **4,202** | 5,678 | 15.9% | compute |
| 2 | 15.20 | 3,618 | 10,691 | 2.3% | compute |
| 4 | 13.41 | 4,102 | 18,444 | 2.3% | compute |
| 8 | 13.75 | 3,999 | 25,601 | 2.1% | compute |

Bottleneck = data loading at 0 workers (≈48% of the epoch spent waiting on data); from 1 worker onwards it is compute (forward/backward/optimizer). Best num_workers = **1** (1.87× faster than 0; 4 is practically tied). Why more workers stopped helping: loader-only throughput keeps scaling (5.3k → 25.6k samples/s), but training stays flat at ~4k samples/s because the model computation is now the slowest part. Extra worker processes also compete with PyTorch's intra-op threads for the same CPU cores, which is the likely cause of the dip at 2 workers and the slight drop at 8. Accuracy was identical across all worker counts (test 0.8768), so this setting only affects speed.

**Batch size** (`results/study_batch.csv`, `.png`):

| batch | samples/s | peak memory (MB) | val acc | test acc (5 ep) | train time (s) |
|---|---|---|---|---|---|
| 32 | 2,002 | 445.4 | 0.9068 | **0.9042** | 133.1 |
| 64 | 2,288 | 444.7 | 0.9004 | 0.8999 | 120.3 |
| 128 | 2,347 | 444.5 | 0.9010 | 0.8958 | 117.2 |
| 256 | 2,472 | 451.3 | 0.8920 | 0.8877 | 111.0 |
| 512 | **2,555** | 489.2 | 0.8780 | 0.8779 | **107.5** |

Throughput rises only 28% from 32 to 512 (the CPU is already busy at small batches), peak memory grows modestly (+44 MB at 512), and test accuracy after 5 epochs falls by 2.6 pp. With a fixed lr of 1e-3 and a fixed 5 epochs, bigger batches mean far fewer optimizer updates (1,719 steps at 32 vs 108 at 512), so they are less trained. Note: peak memory is of the main process only (Windows peak working set; DataLoader workers not included).

**Schedulers / optimizers** (`results/study_schedulers.csv`, `.png`), 10 epochs each:

| run | optimizer | scheduler | best val acc | final val loss | test acc | final lr |
|---|---|---|---|---|---|---|
| sched_step | Adam | StepLR | 0.9082 | 0.2623 | 0.9013 | 1.25e-4 |
| sched_cosine | Adam | Cosine | 0.9092 | 0.2587 | 0.9023 | 2.4e-5 |
| sched_plateau | Adam | ReduceLROnPlateau | 0.9110 | 0.2482 | **0.9033** | 1e-3 (never reduced) |
| opt_adam | Adam | none | 0.9110 | 0.2482 | **0.9033** | 1e-3 |
| opt_sgd_momentum | SGD+m 0.9 | none | 0.9058 | 0.2780 | 0.8986 | 0.05 |

All three schedulers are within 0.2 pp test accuracy of each other. ReduceLROnPlateau never fired in 10 epochs (lr stayed at 1e-3), so its run is identical to plain Adam. StepLR and Cosine decay the lr and give smoother late curves, but they did not beat constant-lr Adam in this short budget. Adam beat SGD+momentum by 0.47 pp test accuracy with lower validation loss; SGD learned quickly early on but was noisier (val acc dropped to 0.893 at epoch 8).

**Accumulation** (`results/study_accum.csv`, `accum_gradient_check.txt`): gradient difference = max |Δgrad| **4.1e-8** (relative **8.9e-7**, largest gradient entry 4.6e-2), so 4×64 accumulated gives the same gradient as one batch of 256 up to float rounding. Curves similar? **Yes**: max |train-loss difference| over 5 epochs = 0.0002, val acc 0.8918 vs 0.8920, test acc 0.8862 vs 0.8877. Throughput was slightly lower (2,832 vs 2,923 samples/s, from more and smaller forward/backward calls), and peak memory was 6 MB lower (444.8 vs 451.1 MB).

**Resume** (`results/study_resume.csv`, `.png`): max difference vs uninterrupted run = **0.0** for both train loss and val acc at every epoch (1–6). Smooth across epoch 3->4? **Yes**: the train-loss step from 3 to 4 is −0.0269 in both runs, and the resumed curve lies exactly on the uninterrupted one. Both runs end at test acc 0.8982. This works because the checkpoint stores model + optimizer state and every epoch reseeds the RNG and sampler. `resume_part1` is logged in MLflow as KILLED (simulated interruption after epoch 3).

**MLflow:** **21 runs** in experiment `lab06` (`results/mlflow_runs.csv`): best_model 1, workers 5, batch 5, schedulers/optimizers 5, accumulation 2, resume 3 (20 FINISHED + 1 KILLED). Each run has `config_used.yaml`, `history.csv` and `summary.json` as artifacts, and 20 also have `model.pt` (not the interrupted part). Screenshot of the MLflow UI comparing runs: **not included in the zip yet; still to add.**

**Proposal:** `lab06/proposal.md`: traffic-sign recognition on GTSRB (43 classes), with knowledge distillation from a ResNet-18 teacher to a small CNN student; target test accuracy ≥ 95%, latency ≤ 50 ms, size ≤ 5 MB (submitted/approved by: ___ ).

## 4. Three conclusions
1. On this CPU-only setup, the first bottleneck was the input pipeline, not the model. With `num_workers=0` about half of each epoch went to loading and transforming images. One background worker removed most of that wait and nearly doubled throughput (2.25k → 4.2k samples/s). After that the model compute is the limit, so adding more workers does nothing.
2. A bigger batch is not automatically better. Going from 32 to 512 increased speed by only 28% and cost 2.6 pp of accuracy under a fixed 5-epoch, fixed-lr budget, because there were far fewer weight updates. For the same reason, the choice of LR schedule mattered very little over 10 epochs (all within 0.2 pp), while the optimizer mattered more (Adam > SGD+momentum).
3. Reproducibility pays off. A single seeded YAML config, full checkpoints and per-epoch reseeding made the interrupted-and-resumed run bit-identical to the uninterrupted one. They also showed that gradient accumulation is mathematically equivalent to a larger batch (relative gradient error ~1e-6), and every run can be traced in MLflow.
