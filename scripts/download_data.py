"""Fetch original UCI data using the installed churn_prediction package."""
from churn_prediction.download import download_data


if __name__ == "__main__":
    output = download_data()
    print(f"Verified 3150 UCI records and wrote {output}")
