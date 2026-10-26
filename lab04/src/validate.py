"""Task 3: schema / range / null-policy / category checks. Reports violations; NEVER drops rows."""
import pandas as pd
from schema import ALL_COLUMNS, EXPECTED_TYPES, RANGES, ALLOWED, MISSING_TOKEN


def validate(df, expected_rows, expected_cols):
    rows = []

    def add(check, col, n, detail, kind="rule"):
        status = ("INFO" if kind == "info" else ("PASS" if n == 0 else "VIOLATION"))
        rows.append({"check": check, "column": col, "n_violations": int(n), "status": status, "detail": detail})

    add("row_count", "*", int(len(df) != expected_rows), f"{len(df)} rows (expected {expected_rows})")
    add("column_count", "*", int(df.shape[1] != expected_cols), f"{df.shape[1]} columns (expected {expected_cols})")
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    add("columns_present", "*", len(missing_cols), f"missing: {missing_cols}" if missing_cols else "all 15 present")

    for col, t in EXPECTED_TYPES.items():
        if col not in df.columns:
            continue
        s = df[col]
        if t == "int":
            if pd.api.types.is_numeric_dtype(s):
                n = int((s.dropna() % 1 != 0).sum())
                add("type_int", col, n, f"dtype {s.dtype}; {n} non-whole values (OpenML stores integers as float)")
            else:
                add("type_int", col, len(s), f"dtype {s.dtype} is not numeric")
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s)
            add("type_category", col, 0 if ok else len(s), f"dtype {s.dtype}")

    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            s = df[col].dropna()
            n = int(((s < lo) | (s > hi)).sum())
            add("range", col, n, f"allowed [{lo}, {hi}]; observed [{s.min():.0f}, {s.max():.0f}]")

    for col in ALL_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        n_nan = int(s.isna().sum())
        n_q = int((s.dropna().astype(str) == MISSING_TOKEN).sum())
        add("missing", col, n_nan + n_q, f"{n_nan} NaN + {n_q} '?' ({100 * (n_nan + n_q) / len(df):.2f}%)", "info")

    for col, allowed in ALLOWED.items():
        if col in df.columns:
            vals = df[col].dropna().astype(str)
            bad = vals[~vals.isin(allowed)]
            detail = "all values allowed" if bad.empty else f"unexpected: {bad.value_counts().head(5).to_dict()}"
            add("allowed_categories", col, len(bad), detail)

    n_dup = int(df.duplicated().sum())
    add("exact_duplicate_rows", "*", n_dup, f"{n_dup} rows repeat an earlier row (kept in report; handled in cleaning)", "info")
    return pd.DataFrame(rows)


def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)
