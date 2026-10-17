# Robustness audit — exact-profile-grouped churn evaluation

This is a **separate sensitivity analysis**, not a replacement for the original
randomly stratified experiment. Identical predictor profiles are kept entirely
within one split and within a single cross-validation fold.

## Source diagnostics

- Total rows: **3,150**
- Unique 11-feature profiles: **2,826**
- Duplicate rows beyond the first: **324**
- Repeated-profile groups: **168**
- Groups with contradictory churn outcomes: **22**

## Group-separated train, validation and test

| Split | Rows | Churn prevalence | Unique profiles |
| --- | ---: | ---: | ---: |
| Train | 1883 | 15.61% | 1695 |
| Validation | 632 | 15.82% | 562 |
| Test | 635 | 15.91% | 569 |

All three pairwise intersections of feature profiles are **zero**.
The model family was selected with stratified **group-aware** training CV.

Selected model: **hist_gradient_boosting**.
Decision threshold: **0.08**, selected on validation F2.

## Test performance (previously untouched, distinct groups)

| Metric | Grouped selected model | Dummy baseline |
| --- | ---: | ---: |
| ROC-AUC | 0.9841 | 0.5000 |
| Average precision | 0.9262 | 0.1591 |
| Precision | 0.5600 | 0.0000 |
| Recall | 0.9703 | 0.0000 |
| F2 | 0.8463 | 0.0000 |
| Accuracy | 0.8740 | 0.8409 |
| Brier score | 0.0357 | 0.1338 |

Confusion matrix [[TN, FP], [FN, TP]]: `[[457, 77], [3, 98]]`.

## Interpretation

This audit excludes exact predictor-profile overlap between splits, but does not
prove there are no duplicate customers with differing measurements. Actual
customer identifiers are not available. It uses an **uncalibrated** classifier
while the original experiment uses a calibrated classifier; differences in
metrics cannot be attributed solely to grouping.

No independent prospective/out-of-time evaluation, intervention uplift study
or fairness audit has been performed. Results are educational, not operational.
