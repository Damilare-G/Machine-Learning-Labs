"""Task 2: profile ONE training epoch with torch.profiler; report share of time per phase."""
import argparse
import csv

import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

import lab3_models
import trainer
from config_utils import load_config

LABELS = ["data_loading", "forward", "backward", "optimizer_step"]
WALL_KEYS = {"data_loading": "data", "forward": "forward", "backward": "backward", "optimizer_step": "optimizer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(trainer.LAB / "config.yaml", a.set)
    dev = trainer.get_device()
    trainer.set_seeds(cfg["seed"])
    train_loader, _, _ = trainer.make_loaders(cfg)
    trainer.set_seeds(cfg["seed"])
    model = lab3_models.build(cfg["model"]).to(dev)
    opt, lossf = trainer.make_optimizer(model, cfg["train"]), nn.CrossEntropyLoss()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])

    with profile(activities=acts) as prof:
        st = trainer.train_epoch(model, train_loader, opt, lossf, dev, int(cfg["train"].get("accum_steps", 1)),
                                 cfg["train"].get("max_batches"))

    ev = {e.key: e for e in prof.key_averages()}
    prof_us = {l: (ev[l].cpu_time_total if l in ev else 0.0) for l in LABELS}       # microseconds, inclusive
    wall = {l: st["phase_s"][WALL_KEYS[l]] * 1000 for l in LABELS}                   # milliseconds
    ptot, wtot = sum(prof_us.values()), sum(wall.values())
    res = trainer.LAB / "results"
    res.mkdir(exist_ok=True)
    rows = [{"phase": l, "profiler_ms": f"{prof_us[l] / 1000:.1f}", "profiler_share_%": f"{100 * prof_us[l] / ptot:.1f}",
             "wall_ms": f"{wall[l]:.1f}", "wall_share_%": f"{100 * wall[l] / wtot:.1f}"} for l in LABELS]
    with open(res / "profile_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (res / "profile_top_ops.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15), encoding="utf-8")
    print(f"model={cfg['model']} workers={cfg['data']['num_workers']} batch={cfg['data']['batch_size']} "
          f"epoch wall={st['epoch_time_s']:.1f}s (profiler overhead included)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()