"""Data contracts and leakage-aware input validation."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "age",
]
CATEGORICAL = ["complains", "age_group", "tariff_plan"]
NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]
TARGET = "churn"
EXCLUDED = ["status", "customer_value"]
REQUIRED = set(FEATURES + EXCLUDED + [TARGET])


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map source headers to snake_case, rejecting collisions."""
    clean = frame.copy()
    clean.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in frame.columns
    ]
    if len(set(clean.columns)) != len(clean.columns):
        raise ValueError("Duplicate normalized column names")
    return clean


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order all 11 approved predictors for model inference."""
    frame = normalize_columns(frame)
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    X = frame.loc[:, FEATURES].copy()
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input contains null or non-finite values")
    if not X["complains"].isin([0, 1]).all():
        raise ValueError("complains must be binary")
    if not X["age_group"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("age_group must be 1..5")
    if not X["tariff_plan"].isin([1, 2]).all():
        raise ValueError("tariff_plan must be 1 or 2")
    if (X[NUMERIC] < 0).any().any():
        raise ValueError("Usage values cannot be negative")
    return X


def load_data(path: str | Path) -> pd.DataFrame:
    """Read UCI source while verifying target, inputs and source metadata."""
    frame = normalize_columns(pd.read_csv(path, encoding="utf-8-sig"))
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    X = validate_input(frame)
    frame.loc[:, FEATURES] = X
    for column in EXCLUDED + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[EXCLUDED + [TARGET]].isna().any().any():
        raise ValueError("Missing outcome or metadata")
    if not frame[TARGET].isin([0, 1]).all() or frame[TARGET].nunique() != 2:
        raise ValueError("Churn labels must contain both 0 and 1")
    return frame
