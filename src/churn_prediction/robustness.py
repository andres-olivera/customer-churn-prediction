"""Leakage-sensitivity audit: identical feature profiles never cross splits.

Unlike the baseline stratified experiment, this audit makes customer-profile
groups indivisible across training, threshold selection and test. Since the UCI
data lacks unique customer IDs, this protects against exact-profile overlap,
but cannot prove that partially matching rows belong to separate customers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .data import FEATURES, TARGET, load_data
from .modeling import SEED, evaluate, models


def feature_profile_groups(frame: pd.DataFrame) -> np.ndarray:
    """One stable group for each exact combination of approved features."""
    index = pd.MultiIndex.from_frame(frame.loc[:, FEATURES])
    codes, _ = pd.factorize(index, sort=False)
    return np.asarray(codes)


def grouped_splits(frame: pd.DataFrame, seed: int = SEED) -> dict:
    """Create disjoint 60/20/20 train/validation/test with group isolation."""
    X = frame[FEATURES].reset_index(drop=True)
    y = frame[TARGET].astype(int).reset_index(drop=True)
    group = feature_profile_groups(X)
    all_idx = np.arange(len(X))
    # One 1/5 grouped fold held out as test; 1/4 of remaining groups as val.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = next(outer.split(X, y, group))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(X.iloc[dev], y.iloc[dev], group[dev]))
    train, val = dev[train_rel], dev[val_rel]
    sets = [set(group[idx]) for idx in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sorted(np.r_[train, val, test].tolist()) == all_idx.tolist()
    return {"X": X, "y": y, "groups": group,
            "train": train, "validation": val, "test": test}


def summarize_duplicate_groups(frame: pd.DataFrame) -> dict:
    groups = feature_profile_groups(frame)
    group_summary = (
        pd.DataFrame({"group": groups, "target": frame[TARGET].astype(int).to_numpy()})
        .groupby("group").agg(size=("target", "size"), unique_targets=("target", "nunique"))
    )
    repeated = group_summary[group_summary["size"] > 1]
    conflicts = group_summary[group_summary["unique_targets"] > 1]
    return {
        "total_rows": int(len(frame)),
        "unique_feature_profiles": int(len(group_summary)),
        "duplicate_rows_beyond_first": int(len(frame) - len(group_summary)),
        "repeated_profile_groups": int(len(repeated)),
        "rows_in_repeated_profile_groups": int(repeated["size"].sum()),
        "conflicting_label_profile_groups": int(len(conflicts)),
        "rows_in_conflicting_label_groups": int(conflicts["size"].sum()),
    }


def audit(data_path, report_path="reports/grouped_holdout.json", cv_folds=5) -> dict:
    """Select and validate a model with exact-feature-profile grouping end-to-end."""
    frame = load_data(data_path)
    split = grouped_splits(frame)
    X, y, g = split["X"], split["y"], split["groups"]
    tr, va, te = split["train"], split["validation"], split["test"]
    candidate = models()
    group_cv = StratifiedGroupKFold(
        n_splits=cv_folds, shuffle=True, random_state=SEED + 2,
    )
    ranking = []
    for name, estimator in candidate.items():
        scoring = cross_validate(
            estimator, X.iloc[tr], y.iloc[tr], groups=g[tr],
            cv=group_cv,
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            error_score="raise", n_jobs=1,
        )
        ranking.append({
            "model": name,
            "cv_average_precision_mean": float(scoring["test_average_precision"].mean()),
            "cv_roc_auc_mean": float(scoring["test_roc_auc"].mean()),
        })
    eligible = [item for item in ranking if item["model"] != "dummy"]
    best_name = max(eligible, key=lambda item: item["cv_average_precision_mean"])["model"]
    best: Pipeline = candidate[best_name].fit(X.iloc[tr], y.iloc[tr])
    valid_scores = best.predict_proba(X.iloc[va])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(max(
        thresholds,
        key=lambda t: fbeta_score(
            y.iloc[va], valid_scores >= t, beta=2, zero_division=0
        ),
    ))
    test_scores = best.predict_proba(X.iloc[te])[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(X.iloc[tr], y.iloc[tr])
    dummy_scores = dummy.predict_proba(X.iloc[te])[:, 1]
    result = {
        "design": "exact-feature-profile grouped split and grouped training CV",
        "dataset": "UCI Iranian Churn (ID 563)",
        "seed": SEED,
        "data_quality": summarize_duplicate_groups(frame),
        "split": {
            key: {
                "rows": int(len(split[key])),
                "churn_rate": float(y.iloc[split[key]].mean()),
                "unique_groups": int(len(set(g[split[key]]))),
            }
            for key in ("train", "validation", "test")
        },
        "train_validation_group_overlap": 0,
        "train_test_group_overlap": 0,
        "validation_test_group_overlap": 0,
        "cv_method": f"StratifiedGroupKFold({cv_folds}), on train only",
        "model_selection_metric": "mean training CV average precision",
        "selected_model": best_name,
        "cv_leaderboard": sorted(
            ranking, key=lambda item: -item["cv_average_precision_mean"]
        ),
        "threshold_selection": "max validation F2; no calibration in grouped audit",
        "validation": evaluate(y.iloc[va], valid_scores, threshold),
        "test": evaluate(y.iloc[te], test_scores, threshold),
        "test_dummy_baseline": evaluate(y.iloc[te], dummy_scores, 0.5),
        "limitations": [
            "Exact matching predictor profiles are proxies, not actual customer IDs.",
            "Dataset lacks observation times, customer IDs and independent cohorts.",
            "Validation and test class prevalences may vary with group sizes.",
            "This uncalibrated model differs from the original calibrated baseline.",
            "The evaluation remains retrospective and cannot estimate intervention uplift.",
        ],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
