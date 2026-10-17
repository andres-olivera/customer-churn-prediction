"""Integrated smoke test with synthetic data only; never infer real-data metrics."""
import numpy as np
import pandas as pd

from churn_prediction.data import FEATURES
from churn_prediction.modeling import train


def test_synthetic_training_smoke(tmp_path):
    rng = np.random.default_rng(7)
    n = 90
    frame = pd.DataFrame({
        name: rng.integers(0, 10, size=n) for name in FEATURES
    })
    frame["complains"] = rng.integers(0, 2, size=n)
    frame["age_group"] = rng.integers(1, 6, size=n)
    frame["tariff_plan"] = rng.integers(1, 3, size=n)
    frame["churn"] = np.asarray([0, 1] * (n // 2))
    frame["status"] = rng.integers(1, 3, size=n)
    frame["customer_value"] = rng.integers(0, 100, size=n)
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    result = train(source, tmp_path / "artifacts", tmp_path / "reports", fast=True)
    assert result["rows_test"] == 18
    assert 0 <= result["test"]["roc_auc"] <= 1
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "churn_model.joblib").exists()
