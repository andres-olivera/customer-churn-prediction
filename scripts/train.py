"""Run from repository root after installing project requirements."""
import argparse
import json

from churn_prediction.modeling import train


def main():
    parser = argparse.ArgumentParser(description="Train calibrated churn classifier")
    parser.add_argument("--data", default="data/Customer_Churn.csv")
    parser.add_argument("--fast", action="store_true", help="3-fold quick smoke check")
    args = parser.parse_args()
    result = train(args.data, fast=args.fast)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "test": result["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
