from __future__ import annotations

import numpy as np
import pandas as pd


def load_raw_data(path) -> pd.DataFrame:
    """Load the original survey workbook and normalize its header."""
    df = pd.read_excel(path, header=3, engine="openpyxl")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic row-level features without learning from the dataset."""
    out = df.copy()
    years_job = out["YearsWithThisTypeOfJob"]
    out["Experience_Level"] = pd.cut(
        years_job,
        bins=[-np.inf, 3, 8, np.inf],
        labels=["Junior", "Mid", "Senior"],
    ).astype("object").fillna("Unknown")
    out["Specialization_Ratio"] = (
        out["YearsWithThisDatabase"] / (years_job + 0.1)
    ).replace([np.inf, -np.inf], np.nan)
    out["Years_Squared"] = years_job.pow(2)
    return out


def clean_domain_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented domain rules only; statistical preprocessing belongs in pipelines."""
    out = df.copy()
    for column in ["YearsWithThisDatabase", "YearsWithThisTypeOfJob"]:
        out.loc[~out[column].between(0, 60), column] = np.nan
    out = out.drop(columns=["Counter"], errors="ignore")
    return add_domain_features(out)


def filter_salary_scope(df: pd.DataFrame, lower: float = 5_000, upper: float = 500_000) -> pd.DataFrame:
    """Restrict modeling to plausible annual salaries using explicit business bounds."""
    return df.loc[df["SalaryUSD"].between(lower, upper)].copy()
