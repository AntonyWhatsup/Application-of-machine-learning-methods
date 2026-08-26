from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .transformers import OtherDatabasesEncoder


def to_string_array(X):
    """Normalize mixed Excel object columns (for example integer/string categories)."""
    return np.asarray(X).astype(str)


def normalize_categorical_missing(X):
    """Represent every pandas/Python missing marker consistently for imputation."""
    if isinstance(X, pd.DataFrame):
        return X.mask(X.isna(), np.nan)
    array = np.asarray(X, dtype=object)
    return pd.DataFrame(array).mask(pd.isna(array), np.nan).to_numpy()


def build_preprocessor(X, *, other_db_min_frequency: float = 0.02) -> ColumnTransformer:
    """Create an unfitted preprocessor. Fit it only through a model Pipeline after splitting."""
    other_db = ["OtherDatabases"] if "OtherDatabases" in X.columns else []
    categorical = [
        column
        for column in X.select_dtypes(include=["object", "category"]).columns
        if column not in other_db
    ]
    numerical = [column for column in X.columns if column not in categorical + other_db]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "normalize_missing",
                FunctionTransformer(
                    normalize_categorical_missing,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(to_string_array, feature_names_out="one-to-one")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers = [
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical),
    ]
    if other_db:
        transformers.append(
            (
                "other_databases",
                OtherDatabasesEncoder(min_frequency=other_db_min_frequency, max_features=30),
                other_db,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
