# Data provenance and usage

**Primary source:** Iranian Churn, UCI Machine Learning Repository (ID 563).
- Dataset: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- DOI: https://doi.org/10.24432/C5JW3Z
- License: Creative Commons Attribution 4.0 (CC BY 4.0); source credits retained.
- Original file: `Customer Churn.csv`, 3,150 records.
- Source ZIP: https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip

The publication states that the predictors summarize the **first nine months** of
customer activity, while churn is measured at the **end of month twelve**.
The three-month gap is a business planning horizon. There is **no** prospective
out-of-time cohort or independent geographic sample.

Run `python scripts/download_data.py` from the project root to fetch the
official ZIP, check CSV headers and row count, and save it locally as
`data/Customer_Churn.csv`. The raw file is gitignored because the pipeline
reproduces it directly from its credited source. Source changes cause verification
to fail rather than silently producing a different experiment.

**Feature exclusions:** `Status` is withheld as a conservative measure because
its semantic proximity to inactivity could mask the actual task; `Customer Value`
is withheld because its underlying formula and operational timing are unclear.

The software is MIT-licensed; **the dataset is CC BY 4.0, not MIT**.
