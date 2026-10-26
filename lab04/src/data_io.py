"""Task 2:"""
import os
import numpy as np
import pandas as pd
from schema import EDU_NUM, ALLOWED


def load_raw(cfg, data_dir):
    """Return the raw OpenML frame exactly as downloaded (14 features + 'class'). Cached under data/openml."""
    if os.environ.get("LAB4_FAKE") == "1":
        return synthetic_adult(cfg["dataset"]["expected_rows"], cfg["seed"])
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(name=cfg["dataset"]["name"], version=cfg["dataset"]["version"],
                         as_frame=True, data_home=str(data_dir / "openml"))
    return bunch.frame


def standardize_columns(df):
    """Only renames OpenML's target column 'class' -> 'income' (values untouched)."""
    return df.rename(columns={"class": "income"})


def synthetic_adult(n=48842, seed=42):
    """Adult-shaped fake data (same columns/dtypes/missing style). NOT for submission."""
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(rng.gamma(4, 9, n) + 17), 17, 90)
    edu_names = np.array(list(EDU_NUM))
    edu_w = np.array([.002, .005, .01, .02, .016, .03, .04, .015, .32, .22, .04, .033, .165, .054, .017, .012])
    edu = rng.choice(edu_names, n, p=edu_w / edu_w.sum())
    edu_num = np.array([EDU_NUM[e] for e in edu], dtype=float)
    sex = rng.choice(["Male", "Female"], n, p=[.67, .33])
    marital = rng.choice(ALLOWED["marital-status"], n, p=[.46, .14, .33, .03, .03, .01, .0])
    others = np.array(["Own-child", "Not-in-family", "Other-relative", "Unmarried"])
    rel = np.where(marital == "Married-civ-spouse", np.where(sex == "Male", "Husband", "Wife"),
                   rng.choice(others, n))
    hours = np.clip(np.round(rng.normal(40, 12, n)), 1, 99)
    gain = np.where(rng.random(n) < .08, np.clip(np.round(rng.lognormal(8.5, 1, n)), 114, 99999), 0)
    loss = np.where(rng.random(n) < .05, rng.integers(1400, 4357, n), 0)
    z = -9.2 + .38 * edu_num + .035 * age + .03 * hours + .0004 * gain + .8 * (sex == "Male") + rng.normal(0, 1, n)
    income = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), ">50K", "<=50K")
    workclass = rng.choice(ALLOWED["workclass"][:7], n, p=[.7, .08, .035, .03, .065, .04, .05]).astype(object)
    workclass[rng.random(n) < .056] = np.nan
    occupation = rng.choice(ALLOWED["occupation"][:-1], n).astype(object)
    occupation[rng.random(n) < .057] = "?"
    country = np.where(rng.random(n) < .9, "United-States", rng.choice(ALLOWED["native-country"][1:-1], n)).astype(object)
    country[rng.random(n) < .017] = np.nan
    df = pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": np.round(rng.lognormal(12, .5, n)),
        "education": edu, "education-num": edu_num, "marital-status": marital, "occupation": occupation,
        "relationship": rel, "race": rng.choice(ALLOWED["race"], n, p=[.85, .03, .01, .01, .10]),
        "sex": sex, "capital-gain": gain.astype(float), "capital-loss": loss.astype(float),
        "hours-per-week": hours, "native-country": country, "class": income})
    # inject: 40 exact duplicates, 5 label conflicts, 2 range violations (so the validator has something to report)
    src = rng.choice(n - 100, 45, replace=False)
    df.iloc[n - 45:n - 5] = df.iloc[src[:40]].to_numpy()
    for k, s in enumerate(src[40:]):
        row = df.iloc[s].copy(); row["class"] = ">50K" if row["class"] == "<=50K" else "<=50K"
        df.iloc[n - 5 + k] = row.to_numpy()
    df.loc[0, "age"] = 5.0
    df.loc[1, "hours-per-week"] = 0.0
    for c in df.columns:
        if c not in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"):
            df[c] = df[c].astype("category")
    for c in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"):
        df[c] = df[c].astype(float)
    return df
