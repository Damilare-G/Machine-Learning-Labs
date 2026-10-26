"""Task 4/5: leakage-free cleaning transformer (fit on TRAIN only) + derived features."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, numeric, categorical, lower_q=0.01, upper_q=0.99, missing_token="?"):
        self.numeric = numeric
        self.categorical = categorical
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.missing_token = missing_token

    def _prep(self, X):
        X = X.copy()
        for c in self.categorical:
            s = X[c].astype(object)
            X[c] = s.where(s != self.missing_token, np.nan)
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, X, y=None):
        X = self._prep(X)
        self.medians_ = {c: float(X[c].median()) for c in self.numeric}
        self.modes_ = {}
        for c in self.categorical:
            m = X[c].mode(dropna=True)
            self.modes_[c] = m.iloc[0] if len(m) else self.missing_token
        filled = X[self.numeric].fillna(self.medians_)
        self.bounds_ = {c: (float(np.quantile(filled[c], self.lower_q)),
                            float(np.quantile(filled[c], self.upper_q))) for c in self.numeric}
        self.n_train_rows_ = len(X)
        return self

    def transform(self, X):
        X = self._prep(X)
        for c in self.numeric:
            X[c] = X[c].fillna(self.medians_[c]).clip(*self.bounds_[c])
        for c in self.categorical:
            X[c] = X[c].fillna(self.modes_[c])
        return X

    def describe_changes(self, X):
        """Counts of imputed and clipped cells if transform() were applied to X (for the report)."""
        P = self._prep(X)
        imputed = int(P[self.numeric + self.categorical].isna().sum().sum())
        clipped = 0
        for c in self.numeric:
            v = P[c].fillna(self.medians_[c])
            lo, hi = self.bounds_[c]
            clipped += int(((v < lo) | (v > hi)).sum())
        return {"imputed_cells": imputed, "clipped_cells": clipped}


def add_derived(X):
    """capital_net = capital-gain - capital-loss ; hours_per_age = hours-per-week / (age - 16)."""
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)   # age >= 17 after clipping
    return X
