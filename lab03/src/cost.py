import csv
import statistics
import time
from pathlib import Path

import torch
from thop import profile
from torchinfo import summary

from common import set_seeds, MODELS
from models import build
import flops_hand

RES = Path(__file__).resolve().parents[1] / "results"
RUNS = {1: 100, 128: 20}


@torch.no_grad()
def latency_ms(model, batch):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    model(x)                                    # 1 warm-up
    ts = []
    for _ in range(RUNS[batch]):
        t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    set_seeds()
    RES.mkdir(exist_ok=True)
    hand = flops_hand.model_totals()
    rows = []
    for name in MODELS:
        m = build(name).eval()
        info = summary(m, input_size=(1, 1, 28, 28), verbose=0)
        (RES / f"summary_{name}.txt").write_text(str(info), encoding="utf-8")
        macs, thop_params = profile(m, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
        rows.append({
            "model": name,
            "params": info.total_params,
            "hand_FLOPs": hand[name],
            "thop_MACs": int(macs),
            "thop_FLOPs(2xMACs)": int(2 * macs),
            "hand/thop": f"{hand[name] / (2 * macs):.2f}",
            "latency_b1_ms": f"{latency_ms(m, 1):.4f}",
            "latency_b128_ms": f"{latency_ms(m, 128):.4f}",
            "threads": torch.get_num_threads(),
        })
        print(rows[-1])
    with open(RES / "cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
