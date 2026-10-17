# Model card — telecom churn alert

## Intended use

Demonstrate transparent supervised classification for a data science portfolio.
A human-reviewed research prototype only; not approved for autonomous retention,
pricing, service denial, profiling or other customer-level consequential decisions.

## Source population

UCI Iranian Churn, 3,150 records collected from one telecom operator; behavior
from months 1–9 and churn at month 12. No independent geographic or future cohort.

## Predictors and model

Eleven approved features: call failures, complaints, subscription length, charge
amount, seconds of use, call frequency, SMS frequency, distinct called numbers,
age group, tariff plan and age. Withheld: Status, Customer Value.

Logistic regression, random forest and histogram gradient boosting compete on
training-only cross-validation. The chosen model is calibrated on training folds;
the decision threshold is determined only from validation. A final holdout is
evaluated once, alongside a constant-prior dummy baseline.

## Known limitations and ethics

- Sensitive demographic signals (age, age group) require fairness auditing,
  privacy assessment and regulatory review before any production use.
- No prospective holdout, randomization, multiple geographies or treatment feedback.
- A high accuracy or ROC-AUC does **not** establish business benefit.
- Interpret model importance as correlations, not reasons customers left.
- Calibration in the source sample is not calibration in another population.
- Retraining and independent out-of-time evaluation would be essential.

## Training evidence

See `reports/EXPERIMENT.md`, `reports/metrics.json` and diagnostic figures
once the verified GitHub workflow has published them.
