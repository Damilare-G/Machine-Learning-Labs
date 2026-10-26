"""Lab 4 pipeline. ONE command from the repo root:  python lab04/src/run_pipeline.py

ingest -> validate -> split (70/30 stratified) -> clean (fit on train only) -> derived features
-> baseline with/without derived -> label audit -> SHA256 + manifest -> data card.
Set LAB4_FAKE=1 for a smoke test on synthetic data (writes to results_smoketest/, never to results/).
"""
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

import audit
import data_card
import model_eval
from data_io import load_raw, standardize_columns
from preprocessing import Preprocessor, add_derived
from schema import NUMERIC, CATEGORICAL, FEATURES, TARGET
from validate import validate, df_to_md

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab04"
log = logging.getLogger("lab04")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_label(y):
    return y.astype(str).str.strip().str.rstrip(".")      # deterministic, no fitted statistics


def build_dataset(raw, cfg):
    """Split FIRST, then dedup train, then fit the preprocessor on train only. Returns everything needed."""
    seed = cfg["seed"]
    X_all, y_all = raw[FEATURES], clean_label(raw[TARGET])
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=cfg["split"]["test_size"],
                                          stratify=y_all, random_state=seed)
    n_before = len(Xtr)
    if cfg["cleaning"]["drop_train_duplicates"]:
        keep = ~pd.concat([Xtr, ytr], axis=1).duplicated()          # exact duplicates, train only
        Xtr, ytr = Xtr[keep], ytr[keep]
    pre = Preprocessor(NUMERIC, CATEGORICAL, cfg["cleaning"]["lower_quantile"],
                       cfg["cleaning"]["upper_quantile"], cfg["cleaning"]["missing_token"]).fit(Xtr)
    changes_train, changes_test = pre.describe_changes(Xtr), pre.describe_changes(Xte)
    Ctr, Cte = add_derived(pre.transform(Xtr)), add_derived(pre.transform(Xte))
    return {"Xtr_raw": Xtr, "Xte_raw": Xte, "ytr": ytr, "yte": yte, "Ctr": Ctr, "Cte": Cte, "pre": pre,
            "train_rows_before_dedup": n_before, "train_duplicates_removed": n_before - len(Xtr),
            "changes_train": changes_train, "changes_test": changes_test}


def write_processed(d, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, X, y in (("train", d["Ctr"], d["ytr"]), ("test", d["Cte"], d["yte"])):
        p = out_dir / f"processed_{name}.csv"
        pd.concat([X, y.rename(TARGET)], axis=1).to_csv(p, index=False, float_format="%.6f")
        paths[name] = p
    return paths


def main():
    cfg = yaml.safe_load(open(LAB / "config.yaml"))
    fake = os.environ.get("LAB4_FAKE") == "1"
    res = LAB / ("results_smoketest" if fake else "results")
    proc = ROOT / "data" / ("processed_smoketest" if fake else "processed")
    res.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(res / "pipeline.log", mode="w")])
    np.random.seed(cfg["seed"])
    if fake:
        log.warning("LAB4_FAKE=1: SYNTHETIC data, outputs go to %s - not valid for submission", res)

    # 2. ingestion
    raw = standardize_columns(load_raw(cfg, ROOT / "data"))
    r, c = raw.shape
    ok = (r, c) == (cfg["dataset"]["expected_rows"], cfg["dataset"]["expected_cols"])
    log.info("INGEST rows=%d (expected %d) cols=%d (expected %d) -> %s", r, cfg["dataset"]["expected_rows"], c,
             cfg["dataset"]["expected_cols"], "OK" if ok else "MISMATCH")

    # 3. validation (reports only; no rows dropped)
    vr = validate(raw, cfg["dataset"]["expected_rows"], cfg["dataset"]["expected_cols"])
    vr.to_csv(res / "validation_report.csv", index=False)
    (res / "validation_report.md").write_text(df_to_md(vr), encoding="utf-8")
    nviol = vr[vr.status == "VIOLATION"]
    log.info("VALIDATE %d checks, %d with violations (%d offending values); rows kept: %d",
             len(vr), len(nviol), int(nviol.n_violations.sum()), len(raw))
    for _, row in nviol.iterrows():
        log.info("  VIOLATION %s [%s]: %d - %s", row["check"], row["column"], row["n_violations"], row["detail"])

    # 4-5. split, clean (train-only fit), derived features
    d = build_dataset(raw, cfg)
    joblib.dump(d["pre"], res / "preprocessor.joblib")
    log.info("SPLIT train=%d test=%d | train exact duplicates removed=%d", d["train_rows_before_dedup"],
             len(d["Xte_raw"]), d["train_duplicates_removed"])
    log.info("CLEAN imputed/clipped cells train=%s test=%s", d["changes_train"], d["changes_test"])
    overlap = int(pd.concat([d["Xte_raw"], d["yte"]], axis=1).merge(
        pd.concat([d["Xtr_raw"], d["ytr"]], axis=1).drop_duplicates(), how="inner").shape[0])
    log.info("test rows that also appear (exactly) in cleaned train: %d", overlap)

    cmp_df = model_eval.compare(d["Ctr"], d["Cte"], d["ytr"], d["yte"], cfg["seed"],
                                cfg["model"]["max_iter"], cfg["positive_label"])
    cmp_df.to_csv(res / "derived_feature_comparison.csv", index=False)
    log.info("BASELINE\n%s", cmp_df.to_string(index=False))

    # 6. label-quality audit (on the raw, full data)
    result, sample = audit.run_audit(raw, cfg["audit"]["sample_size"], cfg["seed"])
    sample.to_csv(res / "audit_sample_50.csv", index=False)
    (res / "label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("AUDIT %s", json.dumps(result))

    # 7. hash + manifest + data card
    paths = write_processed(d, proc)
    h_tr, h_te = sha256_file(paths["train"]), sha256_file(paths["test"])
    combined = hashlib.sha256((h_tr + h_te).encode()).hexdigest()
    log.info("SHA256 train=%s test=%s combined=%s", h_tr, h_te, combined)
    y_full = clean_label(raw[TARGET])
    miss = lambda col: float((raw[col].isna() | (raw[col].astype(object) == "?")).mean())
    stats = {"positive_rate": float((y_full == cfg["positive_label"]).mean()),
             "male_share": float((raw["sex"].astype(object) == "Male").mean()),
             "white_share": float((raw["race"].astype(object) == "White").mean()),
             "us_share": float((raw["native-country"].astype(object) == "United-States").mean()),
             "miss_workclass": miss("workclass"), "miss_occupation": miss("occupation"),
             "miss_country": miss("native-country"),
             "conflicting_groups_14": result["conflicts_all_14_features"]["conflicting_groups"],
             "conflicting_groups_13": result["conflicts_13_features_without_fnlwgt"]["conflicting_groups"]}
    manifest = {
        "config": cfg, "synthetic_data": fake, "raw_rows": r, "raw_cols": c, "seed": cfg["seed"],
        "train_rows_before_dedup": d["train_rows_before_dedup"], "test_rows": len(d["Xte_raw"]),
        "train_duplicates_removed": d["train_duplicates_removed"], "train_rows_final": len(d["Ctr"]),
        "sha256_train": h_tr, "sha256_test": h_te, "sha256_combined": combined,
        "processed_files": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "preprocessor_file": str((res / "preprocessor.joblib").relative_to(ROOT)),
        "metrics": cmp_df.to_dict(orient="records"), "dataset_stats": stats,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyyaml": yaml.__version__},
    }
    (res / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    data_card.write(LAB / ("data_card_smoketest.md" if fake else "data_card.md"), manifest, stats)
    log.info("DONE. Outputs in %s", res)


if __name__ == "__main__":
    main()
