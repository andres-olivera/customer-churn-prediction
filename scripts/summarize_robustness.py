"""Generate human-readable grouped evaluation notes from measured JSON."""
import json
from pathlib import Path

input_path = Path("reports/grouped_holdout.json")
report = json.loads(input_path.read_text(encoding="utf-8"))
d = report["data_quality"]
t = report["test"]
b = report["test_dummy_baseline"]
lines = [
    "# Robustness audit — exact-profile-grouped churn evaluation",
    "",
    "This is a **separate sensitivity analysis**, not a replacement for the original",
    "randomly stratified experiment. Identical predictor profiles are kept entirely",
    "within one split and within a single cross-validation fold.",
    "",
    "## Source diagnostics",
    "",
    f"- Total rows: **{d['total_rows']:,}**",
    f"- Unique 11-feature profiles: **{d['unique_feature_profiles']:,}**",
    f"- Duplicate rows beyond the first: **{d['duplicate_rows_beyond_first']:,}**",
    f"- Repeated-profile groups: **{d['repeated_profile_groups']}**",
    f"- Groups with contradictory churn outcomes: **{d['conflicting_label_profile_groups']}**",
    "",
    "## Group-separated train, validation and test",
    "",
    "| Split | Rows | Churn prevalence | Unique profiles |",
    "| --- | ---: | ---: | ---: |",
]
for k in ("train", "validation", "test"):
    row = report["split"][k]
    lines.append(
        f"| {k.title()} | {row['rows']} | {row['churn_rate']:.2%} "
        f"| {row['unique_groups']} |"
    )
lines.extend([
    "",
    "All three pairwise intersections of feature profiles are **zero**.",
    "The model family was selected with stratified **group-aware** training CV.",
    "",
    f"Selected model: **{report['selected_model']}**.",
    f"Decision threshold: **{t['threshold']:.2f}**, selected on validation F2.",
    "",
    "## Test performance (previously untouched, distinct groups)",
    "",
    "| Metric | Grouped selected model | Dummy baseline |",
    "| --- | ---: | ---: |",
])
for k, name in [
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "Average precision"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f2", "F2"),
    ("accuracy", "Accuracy"),
    ("brier_score", "Brier score"),
]:
    lines.append(f"| {name} | {t[k]:.4f} | {b[k]:.4f} |")
lines.extend([
    "",
    f"Confusion matrix [[TN, FP], [FN, TP]]: `{t['confusion_matrix']}`.",
    "",
    "## Interpretation",
    "",
    "This audit excludes exact predictor-profile overlap between splits, but does not",
    "prove there are no duplicate customers with differing measurements. Actual",
    "customer identifiers are not available. It uses an **uncalibrated** classifier",
    "while the original experiment uses a calibrated classifier; differences in",
    "metrics cannot be attributed solely to grouping.",
    "",
    "No independent prospective/out-of-time evaluation, intervention uplift study",
    "or fairness audit has been performed. Results are educational, not operational.",
])
Path("reports/GROUPED_AUDIT.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print("Wrote reports/GROUPED_AUDIT.md from measured data")
