"""Lab 6 training library: YAML-driven, seeded, checkpointed, MLflow-logged, phase-timed (Tasks 1, 6, 7)."""
import csv
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml
from torch.profiler import record_function
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

import lab3_models
from config_utils import flatten, tracking_uri

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab06"
MEAN, STD = 0.2860, 0.3530


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(cfg):
    """55k train / 5k val from the official train set, official 10k test (same split as Lab 3).
    LAB6_FAKE=1 uses random tensors for quick smoke tests without downloading."""
    seed, d = cfg["seed"], cfg["data"]
    set_seeds(seed)
    if os.environ.get("LAB6_FAKE") == "1":
        fake = lambda n: TensorDataset(torch.randn(n, 1, 28, 28), torch.randint(0, 10, (n,)))
        train, val, test = fake(5500), fake(500), fake(1000)
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
        root = str(ROOT / d["root"])
        full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
        train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(seed))
    nw = int(d["num_workers"])
    train_loader = DataLoader(train, batch_size=int(d["batch_size"]), shuffle=True,
                              num_workers=nw, persistent_workers=nw > 0)
    # evaluation loaders never use workers, so the num_workers study only affects the training loader
    return train_loader, DataLoader(val, batch_size=512), DataLoader(test, batch_size=512)


def make_optimizer(model, t):
    name = t["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t.get("momentum", 0.0),
                               weight_decay=t.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {name}")


def make_scheduler(opt, t, epochs):
    s = str(t.get("scheduler", "none")).lower()
    if s == "none":
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=t.get("step_size", 3), gamma=t.get("gamma", 0.5))
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=t.get("gamma", 0.5),
                                                          patience=t.get("patience", 1))
    raise ValueError(f"unknown scheduler {s}")


def peak_memory_mb():
    """Peak memory of THIS process (worker processes not included). GPU: max allocated; Windows: peak working set."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 2**20
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 1024
    except ImportError:
        return mi.rss / 2**20


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def train_epoch(model, loader, opt, lossf, dev, accum=1, max_batches=None):
    """One epoch with per-phase wall timing and record_function labels (used by the profiler too)."""
    model.train()
    total = len(loader) if not max_batches else min(len(loader), int(max_batches))
    t = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    loss_sum, n, startup = torch.zeros((), device=dev), 0, 0.0
    it, i = iter(loader), 0
    opt.zero_grad(set_to_none=True)
    start = time.perf_counter()
    while i < total:
        t0 = time.perf_counter()
        with record_function("data_loading"):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(dev), y.to(dev)
        t1 = time.perf_counter()
        if i == 0:
            startup = t1 - start
        with record_function("forward"):
            loss = lossf(model(x), y)
            _sync(dev)
        t2 = time.perf_counter()
        with record_function("backward"):
            (loss / accum).backward()          # scale so accumulated grads equal the big-batch mean gradient
            _sync(dev)
        t3 = time.perf_counter()
        if (i + 1) % accum == 0 or i == total - 1:
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
                _sync(dev)
        t4 = time.perf_counter()
        t["data"] += t1 - t0; t["forward"] += t2 - t1; t["backward"] += t3 - t2; t["optimizer"] += t4 - t3
        loss_sum += loss.detach() * len(y)
        n += len(y)
        i += 1
    wall = time.perf_counter() - start
    return {"train_loss": float(loss_sum) / max(n, 1), "samples": n, "epoch_time_s": wall,
            "samples_per_s": n / wall, "phase_s": t, "startup_s": startup}


@torch.no_grad()
def evaluate(model, loader, lossf, dev):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += lossf(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def run(cfg, run_name, out_dir, resume=None, stop_after=None, use_mlflow=True, tags=None):
    t = cfg["train"]
    epochs, accum = int(t["epochs"]), int(t.get("accum_steps", 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    set_seeds(cfg["seed"])
    train_loader, val_loader, test_loader = make_loaders(cfg)
    set_seeds(cfg["seed"])
    model = lab3_models.build(cfg["model"]).to(dev)
    opt, lossf = make_optimizer(model, t), nn.CrossEntropyLoss()
    sched = make_scheduler(opt, t, epochs)
    history, start_epoch, best = [], 1, 0.0

    if resume:                                           # Task 7: restore model, optimizer, scheduler, history
        ck = torch.load(resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])
        history, start_epoch, best = ck["history"], ck["epoch"] + 1, ck["best_val_acc"]
        print(f"RESUMED from {resume} at epoch {start_epoch}")

    ml = None
    if use_mlflow and cfg["mlflow"].get("enabled", True):
        import mlflow
        ml = mlflow
        ml.set_tracking_uri(tracking_uri(cfg, LAB))
        ml.set_experiment(cfg["mlflow"]["experiment"])
        ml.start_run(run_name=run_name)
        ml.set_tags({**(tags or {}), "model": cfg["model"], "device": str(dev), "resumed": str(bool(resume))})
        ml.log_params({**flatten(cfg), "effective_batch": int(cfg["data"]["batch_size"]) * accum})

    interrupted, status = False, "FINISHED"
    try:
        for epoch in range(start_epoch, epochs + 1):
            # per-epoch seeds make shuffling/dropout independent of whether training was interrupted
            set_seeds(cfg["seed"] + epoch)
            train_loader.sampler.generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
            st = train_epoch(model, train_loader, opt, lossf, dev, accum, t.get("max_batches"))
            val_loss, val_acc = evaluate(model, val_loader, lossf, dev)
            lr_used = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step(val_loss) if t["scheduler"] == "plateau" else sched.step()
            row = {"epoch": epoch, "lr": lr_used, "train_loss": st["train_loss"], "val_loss": val_loss,
                   "val_acc": val_acc, "epoch_time_s": st["epoch_time_s"], "samples_per_s": st["samples_per_s"],
                   "data_s": st["phase_s"]["data"], "forward_s": st["phase_s"]["forward"],
                   "backward_s": st["phase_s"]["backward"], "optimizer_s": st["phase_s"]["optimizer"],
                   "startup_s": st["startup_s"]}
            history.append(row)
            is_best = val_acc > best
            best = max(best, val_acc)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict() if sched is not None else None,
                        "history": history, "best_val_acc": best, "cfg": cfg}, out_dir / "last.pt")
            if is_best:
                torch.save(model.state_dict(), out_dir / "best.pt")
            if ml is not None:
                ml.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"[{run_name}] epoch {epoch}/{epochs} lr={lr_used:.5f} train_loss={row['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} {row['samples_per_s']:.0f} samples/s "
                  f"(data {row['data_s']:.1f}s fwd {row['forward_s']:.1f}s bwd {row['backward_s']:.1f}s "
                  f"opt {row['optimizer_s']:.1f}s)", flush=True)
            if stop_after and epoch == stop_after and epoch < epochs:
                interrupted, status = True, "KILLED"
                print(f"INTERRUPTED after epoch {epoch} (simulated); checkpoint: {out_dir / 'last.pt'}")
                break

        test_acc = None
        if not interrupted:
            _, test_acc = evaluate(model, test_loader, lossf, dev)
            torch.save(model.state_dict(), out_dir / "model.pt")
        eps = [h["samples_per_s"] for h in history]
        summary = {
            "run_name": run_name, "model": cfg["model"], "device": str(dev), "interrupted": interrupted,
            "epochs_done": len(history), "final_val_acc": history[-1]["val_acc"],
            "final_val_loss": history[-1]["val_loss"], "final_train_loss": history[-1]["train_loss"],
            "best_val_acc": best, "test_acc": test_acc,
            "throughput_median": statistics.median(eps),
            "throughput_steady": statistics.median(eps[1:]) if len(eps) > 1 else eps[0],
            "peak_memory_mb": peak_memory_mb(), "total_train_s": sum(h["epoch_time_s"] for h in history),
            "history": history, "cfg": cfg,
            "versions": {"python": platform.python_version(), "torch": torch.__version__},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader(); w.writerows(history)
        if ml is not None:
            ml.log_metrics({"final_val_acc": summary["final_val_acc"], "best_val_acc": best,
                            "peak_memory_mb": summary["peak_memory_mb"],
                            "throughput_median": summary["throughput_median"],
                            "total_train_s": summary["total_train_s"],
                            **({"test_acc": test_acc} if test_acc is not None else {})})
            for fn in ("config_used.yaml", "history.csv", "summary.json", "model.pt"):
                if (out_dir / fn).exists():
                    ml.log_artifact(str(out_dir / fn))
        return summary
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if ml is not None:
            ml.end_run(status=status)