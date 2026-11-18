# Individual Project Proposal

**Student name:** Damilare Oyefara    **Student ID:** 73599    **Date:** 18 November 2026

## Problem
Recognise traffic signs from small camera crops on a resource-limited device (for example, a driver-assistance module or a phone app). Predictions must come back fast enough for real-time use, and the model must be small enough to ship on the device. Large CNNs are accurate but too big and too slow, so the goal is a compact model that keeps most of their accuracy.

## Dataset
**GTSRB (German Traffic Sign Recognition Benchmark)**: about 39k training and 12.6k test RGB images across 43 sign classes. It loads with `torchvision.datasets.GTSRB`. Images vary in size, so all are resized to 32×32 and normalised. I will hold out 10% of the training set (seed 42) for validation. Known limits: the images come from German roads only, classes are imbalanced, and many images are consecutive frames of the same sign, so accuracy will overstate real-world performance on other countries and conditions. The dataset is public for research use.

## Success metric
The distilled student model reaches **test accuracy ≥ 95%** on the official GTSRB test set and stays within 2 percentage points of the teacher, measured with a pinned environment and seed 42. It must also meet both deployment constraints below. Macro-F1 is a secondary metric, chosen because of the class imbalance.

## Deployment constraints
- Single-image inference latency **≤ 50 ms** (median of 100 runs after warm-up, batch size 1, on my own CPU).
- Model size **≤ 5 MB** (saved file size).

## Baseline
- **Teacher:** ResNet-18 adapted to 32×32 inputs and 43 classes (about 11M parameters, about 45 MB in float32). Expected to be very accurate, but it breaks the 5 MB limit.
- **Student before distillation:** a small CNN in the style of my Lab 3 model, trained on hard labels only, under the same training budget.

## Planned optimization
**Primary: knowledge distillation.** I will first train the teacher. Then I will train the small student CNN (target under 0.5M parameters, under 2 MB) on a loss that combines cross-entropy with the true labels and a KL-divergence loss on the teacher's softened outputs (temperature T, weight α). I will tune T ∈ {2, 4, 8} and α ∈ {0.5, 0.7, 0.9}. Expected effect: the student meets both limits and scores higher than the same student trained without the teacher. If time allows, I will also try INT8 quantization of the student as a stretch goal.

## Evaluation plan
One table will compare the teacher, the student without distillation, and the distilled student on accuracy, macro-F1, latency, model size and peak memory, all measured the same way. Every run will be tracked in MLflow, with seeds and configs kept in YAML (reusing the Lab 6 pipeline) so each number can be reproduced with one command.

## Risks
The student may not reach 95% without more epochs or augmentation, and the distillation hyperparameters may need more tuning than planned. Latency also depends on the CPU, so all models will be measured on the same machine.
