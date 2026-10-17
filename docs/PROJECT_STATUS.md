# Project status

Code, tests, documentation and workflows are version controlled.

The [verified experiment workflow](../.github/workflows/publish-results.yml)
downloads the official source, trains the models, and commits
`reports/metrics.json`, `reports/EXPERIMENT.md`, and the actual figures.
Do not assume real-data performance until those outputs appear and the run
is successful. No metrics are fabricated.
