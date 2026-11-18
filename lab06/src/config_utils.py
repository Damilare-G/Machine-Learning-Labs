"""Config helpers with no torch dependency."""
from pathlib import Path
import yaml


def load_config(path, overrides=()):
    """Load YAML and apply overrides like 'train.lr=0.01' (values parsed as YAML: numbers, true, null...)."""
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for item in overrides:
        key, val = item.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"unknown config key: {key}")
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        else:
            out[name] = v
    return out


def tracking_uri(cfg, lab_dir):
    path = cfg["mlflow"].get("tracking_dir") or (Path(lab_dir) / "results" / "mlruns")
    return Path(path).resolve().as_uri()