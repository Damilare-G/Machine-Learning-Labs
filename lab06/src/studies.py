"""Tasks 3-7. Every training run is a separate subprocess of train.py ).

  python lab06/src/studies.py workers      # Task 3
  python lab06/src/studies.py batch        # Task 4
  python lab06/src/studies.py schedulers   # Task 5
  python lab06/src/studies.py accum        # Task 6
  python lab06/src/studies.py resume       # Task 7
  python lab06/src/studies.py all
Add --quick for a fast smoke test (30 batches per epoch, fewer epochs).  LAB6_FAKE=1 uses random data.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
LAB = SRC.parent
RES = LAB / "results"
RUNS = RES / "runs"
CONFIG = LAB / "config.yaml"
QUICK = False


def _val(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def run_train(name, sets, tags=None, extra=()):
    sets = dict(sets)
    if QUICK:
        sets["train.max_batches"] = 30
    cmd = [sys.executable, str(SRC / "train.py"), "--run-name", name]
    for k, v in sets.items():
        cmd += ["--set", f"{k}={_val(v)}"]
    for k, v in (tags or {}).items():
        cmd += ["--tag", f"{k}={v}"]
    cmd += list(extra)
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads((RUNS / name / "summary.json").read_text(encoding="utf-8"))


def write_csv(name, rows):
    RES.mkdir(exist_ok=True)
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote results/{name}")
    for r in rows:
        print("  ", r)


def loader_only_throughput(num_workers):
    """Samples/s the DataLoader alone can deliver (no model). Median of 3 passes after a warm-up pass."""
    import trainer
    from config_utils import load_config
    cfg = load_config(CONFIG, [f"data.num_workers={num_workers}"])
    loader, _, _ = trainer.make_loaders(cfg)
    n_batches = 30 if QUICK else len(loader)

    def one_pass(dl):
        n, t0 = 0, time.perf_counter()
        for i, (x, y) in enumerate(dl):
            n += len(y)
            if i + 1 >= n_batches:
                break
        return n / (time.perf_counter() - t0)

    one_pass(loader)
    return statistics.median([one_pass(loader) for _ in range(3)])


# ------------------------------------------------------------------ Task 3
def study_workers():
    rows = []
    for w in [0, 1, 2, 4, 8]:
        s = run_train(f"workers_{w}", {"data.num_workers": w, "train.epochs": 2}, {"study": "workers"})
        first, last = s["history"][0], s["history"][-1]     # epoch 2 = steady state (persistent workers already up)
        phases = last["data_s"] + last["forward_s"] + last["backward_s"] + last["optimizer_s"]
        share = last["data_s"] / phases
        rows.append({"num_workers": w, "epoch1_time_s": round(first["epoch_time_s"], 2),
                     "epoch2_time_s": round(last["epoch_time_s"], 2),
                     "train_samples_per_s": round(last["samples_per_s"], 1),
                     "loader_only_samples_per_s": round(loader_only_throughput(w), 1),
                     "data_wait_share_%": round(100 * share, 1),
                     "bottleneck": "data loading" if share > 0.30 else "compute (fwd/bwd/opt)"})
    write_csv("study_workers.csv", rows)
    xs = [r["num_workers"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, [r["train_samples_per_s"] for r in rows], "o-", label="training (model + loader)")
    ax[0].plot(xs, [r["loader_only_samples_per_s"] for r in rows], "s--", label="loader only")
    ax[0].set(xlabel="num_workers", ylabel="samples / s", title="Throughput vs workers"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar([str(x) for x in xs], [r["data_wait_share_%"] for r in rows])
    ax[1].set(xlabel="num_workers", ylabel="% of epoch waiting on data", title="Data-loading share (epoch 2)")
    fig.tight_layout(); fig.savefig(RES / "study_workers.png", dpi=150)


# ------------------------------------------------------------------ Task 4
def study_batch():
    rows, epochs = [], 2 if QUICK else 5
    for bs in [32, 64, 128, 256, 512]:
        s = run_train(f"batch_{bs}", {"data.batch_size": bs, "train.epochs": epochs}, {"study": "batch"})
        rows.append({"batch_size": bs, "throughput_samples_per_s": round(s["throughput_steady"], 1),
                     "peak_memory_MB": round(s["peak_memory_mb"], 1), "final_val_acc": round(s["final_val_acc"], 4),
                     "test_acc": round(s["test_acc"], 4), "total_train_s": round(s["total_train_s"], 1)})
    write_csv("study_batch.csv", rows)
    xs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, key, lab in zip(ax, ["throughput_samples_per_s", "peak_memory_MB", "test_acc"],
                           ["samples / s", "peak memory (MB)", f"test accuracy after {epochs} epochs"]):
        a.plot(xs, [r[key] for r in rows], "o-"); a.set_xscale("log", base=2)
        a.set(xlabel="batch size", ylabel=lab); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RES / "study_batch.png", dpi=150)


# ------------------------------------------------------------------ Task 5
def study_schedulers():
    epochs = 4 if QUICK else 10
    plans = [("sched_step", "schedulers", {"train.optimizer": "adam", "train.lr": 0.001, "train.scheduler": "step"}),
             ("sched_cosine", "schedulers", {"train.optimizer": "adam", "train.lr": 0.001, "train.scheduler": "cosine"}),
             ("sched_plateau", "schedulers", {"train.optimizer": "adam", "train.lr": 0.001, "train.scheduler": "plateau"}),
             ("opt_adam", "optimizers", {"train.optimizer": "adam", "train.lr": 0.001, "train.scheduler": "none"}),
             ("opt_sgd_momentum", "optimizers", {"train.optimizer": "sgd", "train.lr": 0.05, "train.momentum": 0.9,
                                                 "train.scheduler": "none"})]
    rows, hist = [], {}
    for name, group, sets in plans:
        s = run_train(name, {**sets, "train.epochs": epochs}, {"study": "schedulers"})
        hist[name] = s["history"]
        rows.append({"group": group, "run": name, "optimizer": sets["train.optimizer"],
                     "scheduler": sets["train.scheduler"], "epochs": epochs,
                     "final_val_acc": round(s["final_val_acc"], 4), "best_val_acc": round(s["best_val_acc"], 4),
                     "final_val_loss": round(s["final_val_loss"], 4), "test_acc": round(s["test_acc"], 4),
                     "final_lr": s["history"][-1]["lr"]})
    write_csv("study_schedulers.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hist.items():
        ax[0].plot([r["epoch"] for r in h], [r["val_acc"] for r in h], "o-", label=name)
        ax[1].plot([r["epoch"] for r in h], [r["lr"] for r in h], "o-", label=name)
    ax[0].set(xlabel="epoch", ylabel="validation accuracy", title=f"{epochs}-epoch budget"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set(xlabel="epoch", ylabel="learning rate used", title="LR schedule"); ax[1].set_yscale("log"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RES / "study_schedulers.png", dpi=150)


# ------------------------------------------------------------------ Task 6
def gradient_equivalence():
    """One effective batch of 256: one big step vs 4 accumulated micro-batches of 64 (same weights, same data)."""
    import torch
    import torch.nn as nn
    import lab3_models
    from config_utils import load_config
    cfg = load_config(CONFIG)
    torch.manual_seed(cfg["seed"])
    model = lab3_models.build(cfg["model"]).eval()          # eval: no dropout randomness in the comparison
    lossf = nn.CrossEntropyLoss()
    x, y = torch.randn(256, 1, 28, 28), torch.randint(0, 10, (256,))
    model.zero_grad(); lossf(model(x), y).backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for k in range(4):
        (lossf(model(x[k * 64:(k + 1) * 64]), y[k * 64:(k + 1) * 64]) / 4).backward()
    acc = [p.grad.clone() for p in model.parameters()]
    diff = max((a - b).abs().max().item() for a, b in zip(big, acc))
    scale = max(b.abs().max().item() for b in big)
    msg = (f"max |grad(batch 256) - grad(4 x 64, accumulated)| = {diff:.3e}  (largest gradient entry {scale:.3e})\n"
           f"relative = {diff / scale:.3e}  -> {'equivalent' if diff / scale < 1e-4 else 'CHECK'}")
    (RES / "accum_gradient_check.txt").write_text(msg + "\n")
    print(msg)
    return diff


def study_accum():
    epochs = 2 if QUICK else 5
    diff = gradient_equivalence()
    a = run_train("accum_bs256", {"data.batch_size": 256, "train.accum_steps": 1, "train.epochs": epochs}, {"study": "accum"})
    b = run_train("accum_bs64x4", {"data.batch_size": 64, "train.accum_steps": 4, "train.epochs": epochs}, {"study": "accum"})
    rows = []
    for name, s in (("batch 256, no accumulation", a), ("batch 64 x accumulation 4", b)):
        rows.append({"setup": name, "effective_batch": 256, "final_train_loss": round(s["final_train_loss"], 4),
                     "final_val_acc": round(s["final_val_acc"], 4), "test_acc": round(s["test_acc"], 4),
                     "samples_per_s": round(s["throughput_steady"], 1), "peak_memory_MB": round(s["peak_memory_mb"], 1)})
    rows.append({"setup": "max |train_loss diff| over epochs", "effective_batch": "",
                 "final_train_loss": round(max(abs(x["train_loss"] - y["train_loss"])
                                               for x, y in zip(a["history"], b["history"])), 4),
                 "final_val_acc": round(abs(a["final_val_acc"] - b["final_val_acc"]), 4), "test_acc": "", "samples_per_s": "",
                 "peak_memory_MB": ""})
    write_csv("study_accum.csv", rows)
    plt.figure(figsize=(6, 4))
    for name, s, st in (("batch 256", a, "-"), ("64 x 4 accum", b, "--")):
        plt.plot([h["epoch"] for h in s["history"]], [h["train_loss"] for h in s["history"]], st + "o", label=name)
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend(); plt.grid(alpha=.3)
    plt.title(f"Gradient accumulation (grad diff {diff:.1e})"); plt.tight_layout()
    plt.savefig(RES / "study_accum.png", dpi=150)


# ------------------------------------------------------------------ Task 7
def study_resume():
    total, stop = (4, 2) if QUICK else (6, 3)
    full = run_train("resume_full", {"train.epochs": total}, {"study": "resume"})
    run_train("resume_part1", {"train.epochs": total}, {"study": "resume"}, ["--stop-after-epoch", str(stop)])
    part2 = run_train("resume_part2", {"train.epochs": total}, {"study": "resume"},
                      ["--resume", str(RUNS / "resume_part1" / "last.pt")])
    rows = []
    for f, r in zip(full["history"], part2["history"]):
        rows.append({"epoch": f["epoch"], "uninterrupted_train_loss": round(f["train_loss"], 6),
                     "resumed_train_loss": round(r["train_loss"], 6),
                     "uninterrupted_val_acc": round(f["val_acc"], 4), "resumed_val_acc": round(r["val_acc"], 4),
                     "abs_train_loss_diff": f"{abs(f['train_loss'] - r['train_loss']):.2e}",
                     "abs_val_acc_diff": f"{abs(f['val_acc'] - r['val_acc']):.2e}"})
    write_csv("study_resume.csv", rows)
    fl, rl = [h["train_loss"] for h in full["history"]], [h["train_loss"] for h in part2["history"]]
    print(f"jump across the interruption (epoch {stop}->{stop + 1}): uninterrupted {fl[stop] - fl[stop - 1]:+.4f}, "
          f"resumed {rl[stop] - rl[stop - 1]:+.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, lab in zip(ax, ["train_loss", "val_acc"], ["train loss", "validation accuracy"]):
        a.plot([h["epoch"] for h in full["history"]], [h[key] for h in full["history"]], "o-", label="uninterrupted")
        a.plot([h["epoch"] for h in part2["history"]], [h[key] for h in part2["history"]], "x--", label="interrupted + resumed")
        a.axvline(stop, color="gray", ls=":"); a.set(xlabel="epoch", ylabel=lab); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RES / "study_resume.png", dpi=150)


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["workers", "batch", "schedulers", "accum", "resume", "all"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    QUICK = a.quick
    RES.mkdir(exist_ok=True)
    todo = {"workers": study_workers, "batch": study_batch, "schedulers": study_schedulers,
            "accum": study_accum, "resume": study_resume}
    for name in (todo if a.study == "all" else [a.study]):
        print(f"\n===== study: {name} {'(QUICK)' if QUICK else ''} =====", flush=True)
        todo[name]()


if __name__ == "__main__":
    main()