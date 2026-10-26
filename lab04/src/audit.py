"""Task 6: label-quality audit: duplicate-conflict audit, impossible-value rules, random 50-row sample."""
import numpy as np
import pandas as pd
from schema import FEATURES, EDU_NUM, MARRIED


def duplicate_conflicts(df, feature_cols):
    """Rows with identical features but different income labels."""
    sub = df[df.duplicated(subset=feature_cols, keep=False)]
    if sub.empty:
        return {"duplicate_rows": 0, "duplicate_groups": 0, "conflicting_groups": 0, "rows_in_conflicting_groups": 0}
    g = sub.groupby(feature_cols, observed=True, dropna=False)["income"]
    nuniq, size = g.nunique(), g.size()
    conflict = nuniq > 1
    return {"duplicate_rows": int(len(sub)), "duplicate_groups": int(len(nuniq)),
            "conflicting_groups": int(conflict.sum()), "rows_in_conflicting_groups": int(size[conflict].sum())}


def rule_flags(df):
    """Boolean DataFrame (one column per rule) marking impossible / implausible values."""
    d = df.copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype(object)
    f = pd.DataFrame(index=d.index)
    f["age_out_of_range"] = (d["age"] < 17) | (d["age"] > 90)
    f["hours_out_of_range"] = (d["hours-per-week"] < 1) | (d["hours-per-week"] > 99)
    f["capital_gain_out_of_range"] = (d["capital-gain"] < 0) | (d["capital-gain"] > 99999)
    f["capital_loss_out_of_range"] = (d["capital-loss"] < 0) | (d["capital-loss"] > 4356)
    expected_num = d["education"].map(EDU_NUM)
    f["education_vs_education_num"] = expected_num.notna() & (expected_num != d["education-num"])
    f["husband_female_or_wife_male"] = ((d["relationship"] == "Husband") & (d["sex"] == "Female")) | \
                                       ((d["relationship"] == "Wife") & (d["sex"] == "Male"))
    f["spouse_role_but_not_married"] = d["relationship"].isin(["Husband", "Wife"]) & d["marital-status"].notna() \
                                       & ~d["marital-status"].isin(MARRIED)
    f["never_worked_but_has_occupation"] = (d["workclass"] == "Never-worked") & d["occupation"].notna() \
                                           & (d["occupation"] != "?")
    f["young_with_postgrad_degree"] = (d["age"] <= 20) & d["education"].isin(["Masters", "Prof-school", "Doctorate"])
    f["gain_and_loss_both_positive"] = (d["capital-gain"] > 0) & (d["capital-loss"] > 0)   # suspicious, not impossible
    return f.fillna(False).astype(bool)


def draw_sample(df, n, seed):
    sample = df.sample(n=n, random_state=seed)
    flags = rule_flags(sample)
    out = sample.copy()
    out.insert(0, "row_id", sample.index)
    out["auto_flags"] = flags.apply(lambda r: ";".join(flags.columns[r.to_numpy()]), axis=1)
    out["manual_check_ok"] = ""      # YOU fill: Y / N after checking against the data dictionary
    out["notes"] = ""
    return out


def run_audit(df, sample_size, seed):
    cols13 = [c for c in FEATURES if c != "fnlwgt"]
    result = {
        "conflicts_all_14_features": duplicate_conflicts(df, FEATURES),
        "conflicts_13_features_without_fnlwgt": duplicate_conflicts(df, cols13),
    }
    flags = rule_flags(df)
    result["rule_violations_full_data"] = {c: int(flags[c].sum()) for c in flags.columns}
    sample = draw_sample(df, sample_size, seed)
    result["sample_rows_with_auto_flags"] = int((sample["auto_flags"] != "").sum())
    return result, sample
