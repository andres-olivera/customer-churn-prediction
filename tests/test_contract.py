"""Contract and evaluation unit tests."""
import numpy as np
import pandas as pd
import pytest

from churn_prediction.data import FEATURES, normalize_columns, validate_input
from churn_prediction.modeling import evaluate, models


def sample_row():
    return {name: (1 if name in {"complains", "tariff_plan", "age_group"} else 10)
            for name in FEATURES}


def test_header_normalization():
    data = normalize_columns(pd.DataFrame({"Call  Failure": [3], "Charge  Amount": [1]}))
    assert list(data.columns) == ["call_failure", "charge_amount"]


def test_feature_contract_excludes_leakage_candidates():
    X = validate_input(pd.DataFrame([sample_row()]))
    assert list(X.columns) == FEATURES
    assert "status" not in X and "customer_value" not in X


def test_missing_input_rejected():
    with pytest.raises(ValueError, match="Missing"):
        validate_input(pd.DataFrame({"age": [35]}))


def test_invalid_category_rejected():
    row = sample_row()
    row["tariff_plan"] = 99
    with pytest.raises(ValueError, match="tariff_plan"):
        validate_input(pd.DataFrame([row]))


def test_nonfinite_input_rejected():
    row = sample_row()
    row["age"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_input(pd.DataFrame([row]))


def test_perfect_predictions():
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), 0.5)
    assert result["recall"] == result["precision"] == 1
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_all_models_are_pipelines():
    assert set(models()) == {
        "dummy", "logistic_regression", "random_forest", "hist_gradient_boosting"
    }
