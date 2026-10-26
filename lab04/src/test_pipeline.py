"""Leakage / reproducibility tests.  Run:  pytest lab04/src -q"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data_io import synthetic_adult, standardize_columns
from run_pipeline import build_dataset, write_processed, sha256_file
from preprocessing import Preprocessor
from schema import NUMERIC, CATEGORICAL, FEATURES

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))


@pytest.fixture(scope="module")
def raw():
    return standardize_columns(synthetic_adult(6000, 42))


def test_split_is_disjoint_and_sized(raw):
    d = build_dataset(raw, CFG)
    assert set(d["Xtr_raw"].index).isdisjoint(d["Xte_raw"].index)
    assert abs(len(d["Xte_raw"]) / len(raw) - 0.30) < 0.01


def test_preprocessor_statistics_come_from_train_only(raw):
    d = build_dataset(raw, CFG)
    pre = d["pre"]
    tr = d["Xtr_raw"]
    assert pre.medians_["age"] == pytest.approx(float(tr["age"].median()))
    # stats must differ from those of the full data -> test rows did not influence the fit
    assert pre.medians_["fnlwgt"] != float(pd.to_numeric(raw["fnlwgt"]).median())


def test_changing_test_data_cannot_change_fitted_params(raw):
    a = build_dataset(raw, CFG)
    raw2 = raw.copy()
    raw2.loc[a["Xte_raw"].index, "age"] = 90.0            # corrupt every test row
    b = build_dataset(raw2, CFG)
    assert a["pre"].medians_ == b["pre"].medians_ and a["pre"].bounds_ == b["pre"].bounds_


def test_duplicates_dropped_from_train_only(raw):
    d = build_dataset(raw, CFG)
    assert d["train_duplicates_removed"] > 0
    assert len(d["Xte_raw"]) == len(raw) - d["train_rows_before_dedup"]       # test untouched


def test_no_missing_after_cleaning_and_clipped(raw):
    d = build_dataset(raw, CFG)
    for X in (d["Ctr"], d["Cte"]):
        assert not X.isna().any().any()
        assert not (X[CATEGORICAL] == "?").any().any()
        for c, (lo, hi) in d["pre"].bounds_.items():
            assert X[c].between(lo, hi).all()


def test_hash_is_deterministic(raw, tmp_path):
    hs = []
    for i in range(2):
        d = build_dataset(raw, CFG)
        p = write_processed(d, tmp_path / str(i))
        hs.append((sha256_file(p["train"]), sha256_file(p["test"])))
    assert hs[0] == hs[1]
