"""Task 1 entry point.  python lab06/src/train.py [--set train.lr=0.01 ...] [--resume path] [--stop-after-epoch N]"""
import argparse
import time

import trainer
from config_utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(trainer.LAB / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=0.01")
    ap.add_argument("--tag", action="append", default=[], help="MLflow tag, e.g. study=workers")
    ap.add_argument("--run-name")
    ap.add_argument("--resume", help="checkpoint (last.pt) to resume from")
    ap.add_argument("--stop-after-epoch", type=int, help="simulate an interruption after this epoch")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    name = a.run_name or f"{cfg['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = trainer.LAB / "results" / "runs" / name
    tags = dict(t.split("=", 1) for t in a.tag)
    trainer.run(cfg, name, out, resume=a.resume, stop_after=a.stop_after_epoch,
                use_mlflow=not a.no_mlflow, tags=tags)


if __name__ == "__main__":
    main()