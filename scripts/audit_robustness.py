"""Generate independent, exact-feature-profile-grouped validation metrics."""
import argparse
import json

from churn_prediction.robustness import audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Customer_Churn.csv")
    parser.add_argument("--report", default="reports/grouped_holdout.json")
    args = parser.parse_args()
    summary = audit(args.data, args.report)
    print(json.dumps({
        "selected_model": summary["selected_model"],
        "data_quality": summary["data_quality"],
        "split": summary["split"],
        "test": summary["test"],
    }, indent=2))
