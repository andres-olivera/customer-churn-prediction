# Project status — reproducible portfolio release

**Implementation and real-data experiment: complete.**

- Python CI: passed for Python 3.11 and Python 3.12.
- Real-data experiment workflow: passed on the official UCI dataset (3,150 records).
- Training, test evaluation, experiment summary and five diagnostic images published.
- Selected algorithm: HistGradientBoosting, training-only five-fold CV.
- Final holdout: 630 records, ROC-AUC 0.9878, average precision 0.9411, recall 0.9495.
- [Verified experiment report](../reports/EXPERIMENT.md)
- [Full JSON metrics](../reports/metrics.json)
- [Automated experiment run](https://github.com/andres-olivera/customer-churn-prediction/actions/runs/39028407239)
- [Python CI run](https://github.com/andres-olivera/customer-churn-prediction/actions/runs/39028407278)

The Streamlit application is implemented as a **local demo**, not yet hosted as
a publicly accessible web application. Run the quickstart from the README,
including local model training, before launching Streamlit.

The original source CSV and serialized trained model are deliberately not
committed. Anyone can download the original CC BY 4.0 dataset, rerun the
experiment, and reproduce the saved performance report.

This is a retrospective, research-only demonstration. It has **not** been
independently validated on a later cohort, fairness-audited or approved for
production retention decisions.
