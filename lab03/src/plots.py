"""Task 5: accuracy vs parameters and accuracy vs latency (needs cost.csv and test_accuracy.csv)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    cost = {r["model"]: r for r in csv.DictReader(open(RES / "cost.csv"))}
    acc = {r["model"]: float(r["test_accuracy"]) for r in csv.DictReader(open(RES / "test_accuracy.csv"))}
    panels = [("params", "parameters", 1), ("latency_b1_ms", "latency, batch 1 (ms)", 1),
              ("latency_b128_ms", "latency, batch 128 (ms)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, label, _) in zip(axes, panels):
        for name, r in cost.items():
            ax.scatter(float(r[key]), acc[name], s=80)
            ax.annotate(name, (float(r[key]), acc[name]), textcoords="offset points", xytext=(6, 6))
        ax.set(xlabel=label, ylabel="test accuracy"); ax.grid(alpha=.3)
    fig.suptitle("Fashion-MNIST, 5 epochs: accuracy vs cost"); fig.tight_layout()
    fig.savefig(RES / "tradeoff.png", dpi=150)


if __name__ == "__main__":
    main()
