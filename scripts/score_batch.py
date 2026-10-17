"""Score a batch of 11-feature records with a locally trained, trusted artifact."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn_prediction.data import validate_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV with 11 approved feature columns")
    parser.add_argument("--model", default="artifacts/churn_model.joblib")
    parser.add_argument("--out", default="reports/batch_predictions.csv")
    args = parser.parse_args()
    # joblib unpickles Python objects: NEVER load models from untrusted sources.
    artifact = joblib.load(args.model)
    if not {"model", "threshold", "features"} <= artifact.keys():
        raise ValueError("Invalid model artifact")
    source = pd.read_csv(args.input)
    frame = validate_input(source)
    if list(frame.columns) != artifact["features"]:
        raise ValueError("Artifact feature contract differs from input")
    scores = artifact["model"].predict_proba(frame)[:, 1]
    output = pd.DataFrame({
        "churn_probability": scores,
        "alert": scores >= artifact["threshold"],
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output)} scored rows to {args.out}")


if __name__ == "__main__":
    main()
