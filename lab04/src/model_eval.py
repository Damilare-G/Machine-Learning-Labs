"""Task 5: LogisticRegression baseline with and without the derived features."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from schema import NUMERIC, CATEGORICAL

DERIVED = ["capital_net", "hours_per_age"]


def evaluate(train, test, ytr, yte, use_derived, seed, max_iter, pos_label):
    num = NUMERIC + (DERIVED if use_derived else [])
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pipe = Pipeline([("prep", ct), ("clf", LogisticRegression(max_iter=max_iter, random_state=seed))])
    cols = num + CATEGORICAL
    pipe.fit(train[cols], ytr)                     # scaler/encoder also fit on TRAIN only
    pred = pipe.predict(test[cols])
    return {"features": "with derived" if use_derived else "without derived",
            "n_features_in": len(cols),
            "accuracy": round(accuracy_score(yte, pred), 4),
            "f1_>50K": round(f1_score(yte, pred, pos_label=pos_label), 4)}


def compare(train, test, ytr, yte, seed, max_iter, pos_label):
    rows = [evaluate(train, test, ytr, yte, d, seed, max_iter, pos_label) for d in (False, True)]
    return pd.DataFrame(rows)
