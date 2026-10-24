"""Task 4: train each model 5 epochs, Adam lr=1e-3, batch 128; record val per epoch + final test accuracy."""
import csv
from pathlib import Path
import torch
import torch.nn as nn
from common import set_seeds, get_loaders, MODELS
from models import build

RES = Path(__file__).resolve().parents[1] / "results"
EPOCHS, LR = 5, 1e-3


@torch.no_grad()
def accuracy(model, loader, dev):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    return correct / total


def main():
    RES.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = get_loaders()
    log, final = [], []
    for name in MODELS:
        set_seeds()
        model = build(name).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        for ep in range(1, EPOCHS + 1):
            model.train(); run = n = 0
            for x, y in tr:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
                run += loss.item() * len(y); n += len(y)
            va_acc = accuracy(model, va, dev)
            log.append({"model": name, "epoch": ep, "train_loss": f"{run / n:.4f}", "val_acc": f"{va_acc:.4f}"})
            print(log[-1])
        final.append({"model": name, "test_accuracy": f"{accuracy(model, te, dev):.4f}"})
        print(final[-1])
    for fn, rows in (("train_log.csv", log), ("test_accuracy.csv", final)):
        with open(RES / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()