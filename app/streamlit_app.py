"""Interactive educational demo of calibrated churn risk."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_prediction.data import FEATURES, validate_input

st.set_page_config(page_title="Churn Risk Explorer", page_icon="📈", layout="centered")
st.title("Telecom Customer Churn — Risk Explorer")
st.caption(
    "Portfolio demo only. Scores do not prove retention intervention effectiveness. "
    "Features and predictions require external and fairness validation before any real use."
)

path = Path("artifacts/churn_model.joblib")
if not path.exists():
    st.error("Model not found: run python scripts/download_data.py and python scripts/train.py")
    st.stop()

# Do not load untrusted .joblib files; loading arbitrary pickle can execute code.
@st.cache_resource
def local_model():
    return joblib.load(path)


artifact = local_model()
st.subheader("Subscriber profile")
values = {
    "call_failure": st.number_input("Call failures", min_value=0, value=8),
    "complains": st.selectbox("Complaint reported", [0, 1]),
    "subscription_length": st.number_input("Subscription length (months)", min_value=0, value=20),
    "charge_amount": st.number_input("Charge amount (ordinal 0–9)", min_value=0, max_value=9, value=2),
    "seconds_of_use": st.number_input("Seconds of use", min_value=0, value=3000),
    "frequency_of_use": st.number_input("Calls", min_value=0, value=40),
    "frequency_of_sms": st.number_input("SMS frequency", min_value=0, value=20),
    "distinct_called_numbers": st.number_input("Distinct called numbers", min_value=0, value=15),
    "age_group": st.selectbox("Age group (encoded 1–5)", [1, 2, 3, 4, 5], index=2),
    "tariff_plan": st.selectbox("Tariff plan", [1, 2]),
    "age": st.number_input("Age", min_value=0, max_value=120, value=35),
}
if st.button("Predict risk", type="primary"):
    X = validate_input(pd.DataFrame([values]))
    if list(X.columns) != artifact["features"]:
        st.error("Model feature contract mismatch; retrain the model.")
        st.stop()
    probability = float(artifact["model"].predict_proba(X[FEATURES])[0, 1])
    flagged = probability >= artifact["threshold"]
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.write("⚠️ Above alert threshold" if flagged else "Below alert threshold")
    st.caption(f"Alert threshold chosen on validation: {artifact['threshold']:.2f}")
