"""Fold-contained preprocessing, model comparison and locked holdout evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, TARGET, load_data

SEED = 42


def preprocessor() -> ColumnTransformer:
    """All learned transformations are inside training folds."""
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)


def models() -> dict[str, Pipeline]:
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=1, random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED,
        ),
    }
    return {
        name: Pipeline([("preprocessor", preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def evaluate(y_true, probabilities, threshold: float) -> dict:
    scores = np.asarray(probabilities)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train(data_path, output_dir="artifacts", reports_dir="reports", fast=False) -> dict:
    frame = load_data(data_path)
    X, y = frame[FEATURES].copy(), frame[TARGET].astype(int)
    # First split isolates a test set which is never used for fitting or tuning.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED,
    )
    candidates = models()
    cv = StratifiedKFold(n_splits=(3 if fast else 5), shuffle=True, random_state=SEED)
    leaderboard = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            error_score="raise", n_jobs=1,
        )
        leaderboard.append({
            "model": name,
            "cv_average_precision_mean": float(scores["test_average_precision"].mean()),
            "cv_average_precision_std": float(scores["test_average_precision"].std()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        })
    # Dummy prior is always benchmark only, never selected for the deliverable.
    eligible = [row for row in leaderboard if row["model"] != "dummy"]
    selected = max(eligible, key=lambda row: row["cv_average_precision_mean"])["model"]
    calibrated = CalibratedClassifierCV(
        estimator=candidates[selected], method="sigmoid", cv=3,
    )
    calibrated.fit(X_train, y_train)
    val_proba = calibrated.predict_proba(X_val)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(max(
        grid,
        key=lambda t: fbeta_score(y_val, val_proba >= t, beta=2, zero_division=0),
    ))
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    dummy = candidates["dummy"].fit(X_train, y_train)
    baseline = evaluate(y_test, dummy.predict_proba(X_test)[:, 1], 0.5)
    result = {
        "dataset": "UCI Iranian Churn (ID 563)",
        "seed": SEED,
        "rows_total": len(X),
        "rows_train": len(X_train),
        "rows_validation": len(X_val),
        "rows_test": len(X_test),
        "prevalence": float(y.mean()),
        "selected_model": selected,
        "features": FEATURES,
        "excluded_features": EXCLUDED,
        "selection": "training-only CV average precision",
        "threshold_selection": "validation-only F2",
        "calibration": "training-only sigmoid 3-fold CV",
        "cv_leaderboard": sorted(
            leaderboard, key=lambda row: -row["cv_average_precision_mean"]
        ),
        "validation": evaluate(y_val, val_proba, threshold),
        "test": evaluate(y_test, test_proba, threshold),
        "test_dummy_baseline": baseline,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": calibrated, "threshold": threshold,
        "features": FEATURES, "selected_model": selected,
    }, Path(output_dir) / "churn_model.joblib")
    (Path(reports_dir) / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result
