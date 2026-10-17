# Customer Churn Prediction

[![CI](https://github.com/andres-olivera/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/andres-olivera/customer-churn-prediction/actions/workflows/ci.yml)
[![Verified experiment](https://github.com/andres-olivera/customer-churn-prediction/actions/workflows/publish-results.yml/badge.svg)](https://github.com/andres-olivera/customer-churn-prediction/actions/workflows/publish-results.yml)

**An end-to-end machine learning project for telecom customer churn.**

**Python · scikit-learn · cross-validation · calibrated classification · Streamlit · GitHub Actions**

Built to show an analytical workflow beyond dashboarding: formulate a business problem, enforce data quality, prevent train/test leakage, compare multiple models, select a decision threshold, and report honest, reproducible performance.

> The dataset describes customer behavior across months 1–9 and whether the account churned by month 12. This is a three-month planning horizon, **not** a real-time forecast from live production data.

## Architecture

```mermaid
flowchart LR
    A[UCI Iranian Churn data] --> B[Validate dataset]
    B --> C[Stratified 60 / 20 / 20 split]
    C --> D[5-fold CV on train only]
    D --> E[Select model by average precision]
    E --> F[Training-only sigmoid calibration]
    F --> G[Choose F2 threshold on validation]
    G --> H[One untouched test evaluation]
    H --> I[JSON metrics and diagnostic plots]
    F --> J[Local model artifact]
    J --> K[Streamlit and batch scoring]
```

## Dataset

[UCI Iranian Churn (ID 563)](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) — 3,150 customers from one telecom operator. Source data is licensed under **CC BY 4.0**; the software in this repository uses an MIT license. Original file is downloaded from UCI and intentionally excluded from Git. See [data provenance](data/README.md).

Our classifier uses 11 predictors including call failures, complaints, length of subscription, call/SMS usage and tariff. It **excludes** `Status` as a conservative leakage precaution and `Customer Value` because its derivation and availability are not documented in sufficient detail for inference.

## What is evaluated?

| Stage | Method |
| --- | --- |
| Split | Stratified 60% training / 20% validation / 20% **untouched test** |
| Models | Logistic regression, random forest, histogram gradient boosting |
| Reference | Constant-prior dummy classifier |
| Model selection | Training-only 5-fold CV, maximize average precision |
| Calibration | 3-fold sigmoid calibration on training only |
| Threshold selection | Maximize F2 on validation, emphasizing recall |
| Test metrics | ROC-AUC, average precision, precision, recall, F2, Brier score, confusion matrix |
| Interpretation | Permutation importance on the locked test set (descriptive, not causal) |

All training transformations are estimated within CV folds. Test data is not used to choose the algorithm or threshold.

## Results

**Verified experiment on the original UCI dataset (3,150 customers).** The pipeline selected **HistGradientBoosting** by cross-validated average precision. The decision threshold (0.27) was tuned on validation data, not on the 630-customer test holdout.

| Metric | Untouched test result |
| --- | ---: |
| ROC-AUC | **0.9878** |
| Average precision (PR-AUC summary) | **0.9411** |
| Accuracy | **95.40%** |
| Precision (at threshold 0.27) | **79.66%** |
| Recall (at threshold 0.27) | **94.95%** |
| F2 score | **0.9144** |
| False positives / missed churners | **24 / 5** |

The model correctly flagged 94 of 99 churners in the test split, at the cost of 24 false alerts. **These figures are retrospective on one operator, not independently validated production performance.**

![ROC curve](reports/figures/roc_curve.png)

![Precision-recall curve](reports/figures/precision_recall.png)

## Robustness check: identical customer profiles

A second, independently published sensitivity audit groups rows with **identical
11-feature profiles** so that no such profiles can cross training,
validation, test or cross-validation folds.

- **2,826 unique profiles** among 3,150 rows; 324 repeat rows beyond the first.
- **168 repeated-profile groups**, including **22 groups** with conflicting churn labels.
- Group-isolated holdout (**635 rows**): **ROC-AUC 0.9841**, **average precision 0.9262**.
- At validation-selected threshold **0.08**: precision **56.0%**, recall **97.0%**,
  with 77 false positives and 3 missed churners.

This check suggests ranking is robust to *exact-profile overlap*. The grouped
audit uses an **uncalibrated** model, so its threshold-based performance is **not
directly comparable** to the original calibrated experiment. Neither test proves
future performance or effectiveness of retention interventions.

Read the **[group-aware robustness audit](reports/GROUPED_AUDIT.md)**,
[measured JSON](reports/grouped_holdout.json) and
[methodological discussion](docs/METHODOLOGY.md).

**All experiment outputs:**

- [Experiment summary](reports/EXPERIMENT.md) — human-readable, versioned once training succeeds.
- [Full performance and cross-validation comparison](reports/metrics.json).
- [Diagnostic figures](reports/figures).

Reports are published by the [Verified experiment workflow](.github/workflows/publish-results.yml) after downloading and validating the original source, running tests, fitting models, and generating charts. **No placeholder metric values** are presented as empirical findings. If links above are missing, the real-data experiment has not yet published successfully.

## Local quick start

Requires Python 3.11 or 3.12, with commands run from the repository root:

```bash
git clone https://github.com/andres-olivera/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,app]"
pytest -q
python scripts/download_data.py
python scripts/train.py
python scripts/figures.py
python scripts/summarize_results.py
streamlit run app/streamlit_app.py
```

To score a CSV with the 11 documented feature columns:

```bash
python scripts/score_batch.py path/to/new_customers.csv --out reports/batch_predictions.csv
```

Run `python scripts/train.py --fast` only for a quicker engineering check; use the **default full procedure** for portfolio results.

## Repository structure

```text
├── .github/workflows/         CI and verified real-data evaluation
├── app/streamlit_app.py       Single-customer risk exploration
├── data/README.md            Source provenance and licensing
├── docs/                     Methodology, business case, model card
├── scripts/                  Download, train, report, batch score
├── src/churn_prediction/     Reusable modeling and data modules
├── tests/                    Unit and end-to-end synthetic tests
├── reports/                  Verified results generated by workflow
└── pyproject.toml            Reproducible Python project metadata
```

## Responsible use and limitations

This is a **portfolio prototype**, not an operational system. A random split of a single historic operator dataset cannot establish future generalization, validate demographic fairness, measure retention campaign uplift, or determine incremental profit. Age and age-group fields require a legal and fairness review before real-world customer decisions. Predictor importance is not causal. Serialized joblib files should never be opened from untrusted sources.

See [methodology](docs/METHODOLOGY.md), [model card](docs/MODEL_CARD.md) and [business framing](docs/BUSINESS_CASE.md).

**Author:** Andrés Olivera Domínguez · [GitHub profile](https://github.com/andres-olivera)
