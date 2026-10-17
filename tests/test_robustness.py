"""Exact-profile group integrity and data-quality regression tests."""
import numpy as np
import pandas as pd

from churn_prediction.data import FEATURES
from churn_prediction.robustness import (
    feature_profile_groups, grouped_splits, summarize_duplicate_groups,
)


def make_sample():
    rng = np.random.default_rng(27)
    n = 220
    base = pd.DataFrame({
        name: rng.integers(1, 40, size=n) for name in FEATURES
    })
    base["complains"] = rng.integers(0, 2, size=n)
    base["age_group"] = rng.integers(1, 6, size=n)
    base["tariff_plan"] = rng.integers(1, 3, size=n)
    base["churn"] = np.tile([0, 1], n // 2)
    # Append identical feature profiles including one with a conflicting label.
    repeats = base.iloc[:8].copy()
    repeats.loc[repeats.index[0], "churn"] = 1 - repeats.iloc[0]["churn"]
    return pd.concat([base, repeats], ignore_index=True)


def test_exact_profiles_share_group():
    frame = make_sample()
    ids = feature_profile_groups(frame)
    assert ids[0] == ids[220]
    assert ids[7] == ids[227]
    assert len(set(ids)) == 220


def test_grouped_splits_no_overlap():
    frame = make_sample()
    splits = grouped_splits(frame)
    ids = splits["groups"]
    a, b, c = [set(ids[splits[k]]) for k in ("train", "validation", "test")]
    assert not (a & b or a & c or b & c)
    assert sum(len(splits[k]) for k in ("train", "validation", "test")) == len(frame)


def test_duplicate_audit_conflicting_labels():
    stats = summarize_duplicate_groups(make_sample())
    assert stats["duplicate_rows_beyond_first"] == 8
    assert stats["repeated_profile_groups"] == 8
    assert stats["conflicting_label_profile_groups"] == 1
