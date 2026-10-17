# Customer Churn Prediction — case study for hiring managers

**[Launch the public interactive Streamlit demo](https://andres-churn-prediction.streamlit.app/)**
(manual live smoke checks on 17 October 2026: model initialization, fictional
single-customer scoring, a two-row fictional CSV upload and scoring, and
interactive changes to a hypothetical retention economics scenario).
This demonstration does not establish real-world churn uplift or production fitness.

## The question

Could a telecom company prioritize account reviews based on estimated
churn risk from historical behavior data three months before the outcome?

## Dataset and pipeline

Public UCI Iranian Churn; 3,150 records with 11 selected model inputs
from months 1–9 and outcome at month 12. Original data is credited and
retrieved reproducibly; ambiguous `Status` and `Customer Value` are
excluded from the model.

End-to-end work: schema checks, fold-contained preprocessing,
cross-validated selection across three classifiers versus a dummy baseline,
calibration, validation-only threshold selection, locked test performance,
automated verification and Streamlit input UI.

## What the results actually show

**Original stratified evaluation** on 630 held-out records:

- HistGradientBoosting, test ROC-AUC **0.9878**, average precision **0.9411**.
- Threshold 0.27 selected on validation: **94 / 99** true churners
  detected, 24 false positives.

**Exact-feature-profile-grouped sensitivity analysis** on 635 distinct-profile
test records:

- 3,150 rows contain **2,826** unique feature profiles; **324** extra
  rows duplicate another predictor profile.
- Stratified group-aware train/validation/test and grouped CV prevent
  exact profiles appearing across partitions.
- ROC-AUC **0.9841**, average precision **0.9262**.
- Validation-selected threshold 0.08: **98 / 101** detected churners,
  **77** false positives (precision 56.0%).
- This sensitivity model is not calibrated like the original, so differences
  in threshold metrics are not attributable only to grouping.

Detailed comparison and limitations:
[original experiment](../reports/EXPERIMENT.md),
[grouped audit](../reports/GROUPED_AUDIT.md),
[model card](MODEL_CARD.md).

## What I would explain in an interview

**Why average precision?** Churn is a minority event; AP directly
summarizes ranking of positive examples, with ROC-AUC as a second view.

**Why three splits?** Cross-validation is restricted to train, while
validation chooses an alert threshold, leaving test untouched.

**Why audit profiles?** Exact feature matches could otherwise cross
random partitions and exaggerate generalization.

**Why can recall and precision trade off?** A lower threshold flags more
customers, catching additional churners at the expense of contact workload.

**Why can't we claim ROI?** Observational churn labels provide no
randomized treatment effect or cost information.

**What would I do in production?** Obtain a prospective cohort with
real customer IDs, assess fairness and drift, and run a randomized
intervention study before launching a retention policy.

## Accurate CV bullets (English)

- Developed and evaluated a calibrated telecom churn classifier in
  scikit-learn with cross-validated model selection and a locked test set;
  achieved **0.988 ROC-AUC** on the original 630-record retrospective holdout.
- Audited duplicate feature profiles, re-evaluated with **group-isolated**
  train/test partitions and achieved **0.984 ROC-AUC** on a distinct-profile
  holdout, with explicit false-alert trade-offs.
- Automated source validation, reporting and tests with GitHub Actions and
  built a Streamlit demonstration supporting fictional individual and CSV scoring.

## Limitations

No independent future customer cohort, verified customer identifiers,
causal retention uplift or fairness clearance. The public web demo
is for fictional educational use only. This is a portfolio experiment,
**not** a production service.
