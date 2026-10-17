"""Export reproducible holdout diagnostics and EDA plots."""
from churn_prediction.reporting import generate_figures

if __name__ == "__main__":
    generate_figures("data/Customer_Churn.csv")
