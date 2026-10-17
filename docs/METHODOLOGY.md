# Methodology: leakage-aware supervised learning

## Problem and timing

Predict a binary churn outcome at month 12 from behavior aggregated during months 1–9.
The evaluation is retrospective. There is no customer identifier, prospective test
cohort or way to identify repeat observations from the provided data.

## Validation design

1. Normalize headers and validate feature and label contracts.
2. Exclude `Status` and `Customer Value` from the predictor set.
3. Split with seed 42: 60% training, 20% validation, 20% **untouched test**.
4. Run 5-fold stratified CV on training rows, with imputation, scaling and encoding
   each fit **inside each fold**.
5. Evaluate four estimators (including dummy reference). Among three predictive
   candidates choose the one with highest mean training CV average precision.
6. Fit a sigmoid-calibrated version using only training data and 3-fold internal CV.
7. Select classification threshold maximizing **F2** on validation holdout.
8. Evaluate on the untouched test once; publish accuracy, precision, recall, F2,
   average precision, ROC-AUC, Brier score and confusion matrix.
9. Generate ROC, precision-recall, confusion matrix, class distribution and
   test-set permutation importance figures. Importance is descriptive only
   and **must not be used for tuning**, which would leak test information.

## Interpreting the results

Average precision is appropriate for comparing ranking performance with a
minority positive class; ROC-AUC provides complementary discrimination quality.
F2 places four times as much weight on recall as precision at the decision
threshold, but it is **not an economic optimum**. Threshold and expected intervention
value should be re-estimated under actual business costs and holdout campaign data.
Sigmoid calibration is not proof that probabilities transfer across populations.

## Important limitations

This design avoids conventional train/test leakage but still uses a **random
retrospective split**. It cannot test changes over time, new customers, cross-company
transfer, treatment effect, identity duplication or deployment drift.
Age/age-group may cause disparate errors; deploy only after a fairness and
legal review. Dataset-specific associations are not causal explanations.
Never load a joblib/pickle model from untrusted input.

## Reproduction

`python scripts/download_data.py` then `python scripts/train.py`,
`python scripts/figures.py`, `python scripts/summarize_results.py`.
All reported test scores originate in `reports/metrics.json` from that run.
