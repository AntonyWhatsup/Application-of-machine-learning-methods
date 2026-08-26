from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OtherDatabasesEncoder(BaseEstimator, TransformerMixin):
    """Encode a comma-separated database list as train-fitted multi-label indicators."""

    def __init__(self, min_frequency: float = 0.02, max_features: int = 30):
        self.min_frequency = min_frequency
        self.max_features = max_features

    @staticmethod
    def _series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        array = np.asarray(X)
        return pd.Series(array.ravel())

    @staticmethod
    def _tokens(value) -> set[str]:
        if pd.isna(value):
            return set()
        return {part.strip() for part in str(value).split(",") if part.strip()}

    def fit(self, X, y=None):
        rows = [self._tokens(value) for value in self._series(X)]
        counts = Counter(token for row in rows for token in row)
        threshold = max(1, int(np.ceil(len(rows) * self.min_frequency)))
        candidates = [item for item in counts.most_common() if item[1] >= threshold]
        self.classes_ = np.array([name for name, _ in candidates[: self.max_features]], dtype=object)
        return self

    def transform(self, X):
        rows = [self._tokens(value) for value in self._series(X)]
        return np.asarray([[token in row for token in self.classes_] for row in rows], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"OtherDatabases__{name}" for name in self.classes_], dtype=object)
