"""Deployable public-facing, educational churn risk explorer.

Run locally with: streamlit run app/streamlit_app.py
For Streamlit Community Cloud, requirements.txt installs the app dependencies.
Model and source data are generated on the server on first initialization.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_prediction.data import FEATURES, validate_input

st.set_page_config(page_title="Customer Churn | ML Portfolio", page_icon="📊", layout="wide")
ARTIFACT = Path("artifacts/churn_model.joblib")
DATA = Path("data/Customer_Churn.csv")


@st.cache_resource(show_spinner="Training and loading the UCI model…")
def get_or_train_model():
    """Only load models created from this repository's own trusted training code."""
    if not ARTIFACT.exists():
        if not DATA.exists():
            from scripts.download_data import main as fetch_original

            fetch_original()
        from churn_prediction.modeling import train

        train(str(DATA), output_dir=str(ARTIFACT.parent), reports_dir="reports")
    result = joblib.load(ARTIFACT)
    if result["features"] != FEATURES:
        raise ValueError("Model input contract mismatch; retraining is required.")
    return result


def score(model_record, frame: pd.DataFrame) -> pd.DataFrame:
    """Never include hidden target, customer value or Status in the predictors."""
    X = validate_input(frame)
    probability = model_record["model"].predict_proba(X[FEATURES])[:, 1]
    return pd.DataFrame({
        "churn_probability": probability.round(5),
        "above_validation_threshold": probability >= model_record["threshold"],
    })


st.title("Customer Churn | Machine Learning Explorer")
st.write(
    "A reproducible portfolio demonstration: predict churn from historical "
    "telecom behavior, compare customer-risk scores, and understand model limitations."
)
st.info(
    "Research prototype only. **Do not upload real personal/customer data**. "
    "Use invented records for demonstrations. Scores are not approved for "
    "consequential decisions, and cannot measure a campaign's causal impact."
)
st.caption(
    "Source: UCI Iranian Churn Dataset (CC BY 4.0). Predictors describe "
    "months 1–9; label is churn at month 12. "
    "Training excludes Status and Customer Value."
)

if not ARTIFACT.exists() and not st.session_state.get("initialize_demo", False):
    st.warning(
        "No local model artifact yet. Initialize the demonstration to download "
        "the public UCI dataset and train the selected classifier. "
        "The first initialization may take a few minutes."
    )
    if st.button("Initialize ML demonstration", type="primary"):
        st.session_state["initialize_demo"] = True
        st.rerun()
    st.stop()

try:
    record = get_or_train_model()
except Exception as exc:
    st.error(f"Model initialization failed: {exc}")
    st.info("Please check network access to UCI and the training environment.")
    st.stop()

with st.sidebar:
    st.header("Experiment context")
    st.metric("Decision threshold", f"{record['threshold']:.2f}")
    st.write(f"Trained model: **{record.get('selected_model', 'classifier')}**")
    st.caption(
        "The threshold was selected on a held-out validation set to maximize F2; "
        "it does **not** represent an economically optimal campaign policy."
    )
    st.markdown(
        "[Read original experiment](https://github.com/andres-olivera/"
        "customer-churn-prediction/blob/main/reports/EXPERIMENT.md)  \\n"
        "[Read robustness audit](https://github.com/andres-olivera/"
        "customer-churn-prediction/blob/main/reports/GROUPED_AUDIT.md)"
    )

tab_single, tab_batch, tab_method = st.tabs(
    ["Single fictional subscriber", "Batch CSV scoring", "Limitations & documentation"]
)

with tab_single:
    st.subheader("Enter a fictional customer")
    with st.form("individual_customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            call_failure = st.number_input("Call failures", min_value=0, value=8)
            complains = st.selectbox("Complaints", [0, 1])
            subscription_length = st.number_input(
                "Subscription length (months)", min_value=0, value=20
            )
            charge_amount = st.number_input(
                "Charge amount, encoded 0–9", min_value=0, max_value=9, value=2
            )
        with c2:
            seconds_of_use = st.number_input("Seconds of use", min_value=0, value=3000)
            frequency_of_use = st.number_input("Call frequency", min_value=0, value=40)
            frequency_of_sms = st.number_input("SMS frequency", min_value=0, value=20)
        with c3:
            distinct_called_numbers = st.number_input(
                "Distinct called numbers", min_value=0, value=15
            )
            age_group = st.selectbox("Encoded age group", [1, 2, 3, 4, 5], index=2)
            tariff_plan = st.selectbox("Tariff plan", [1, 2])
            age = st.number_input("Age (fictional)", min_value=0, max_value=120, value=35)
        submitted = st.form_submit_button("Estimate churn risk", type="primary")
    row = {
        "call_failure": call_failure,
        "complains": complains,
        "subscription_length": subscription_length,
        "charge_amount": charge_amount,
        "seconds_of_use": seconds_of_use,
        "frequency_of_use": frequency_of_use,
        "frequency_of_sms": frequency_of_sms,
        "distinct_called_numbers": distinct_called_numbers,
        "age_group": age_group,
        "tariff_plan": tariff_plan,
        "age": age,
    }
    if submitted:
        try:
            result = score(record, pd.DataFrame([row]))
            prob = float(result.iloc[0]["churn_probability"])
            st.metric("Estimated probability of churn", f"{prob:.1%}")
            st.write(
                "Above validation alert threshold"
                if bool(result.iloc[0]["above_validation_threshold"])
                else "Below validation alert threshold"
            )
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(
                "Illustrative score only. Not a diagnosis, causal explanation or "
                "recommendation for an actual customer."
            )
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")

with tab_batch:
    st.subheader("Try a small fictional CSV")
    sample = pd.DataFrame([row, {**row, "complains": 1, "frequency_of_sms": 0}])
    st.download_button(
        "Download fictional CSV template",
        data=sample[FEATURES].to_csv(index=False),
        file_name="fictional_customers.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Upload a fictional CSV", type="csv", key="batch_input")
    if upload is not None:
        if upload.size > 2_000_000:
            st.error("Demo limit: upload a CSV smaller than 2 MB.")
        else:
            try:
                uploaded = pd.read_csv(upload)
                if uploaded.empty or len(uploaded) > 5000:
                    raise ValueError("Provide between 1 and 5,000 rows")
                scored = score(record, uploaded)
                st.metric("Fictional rows scored", f"{len(scored):,}")
                st.metric(
                    "Above validation threshold",
                    f"{int(scored['above_validation_threshold'].sum()):,}",
                )
                st.bar_chart(
                    scored["churn_probability"].value_counts(
                        bins=[0, .2, .4, .6, .8, 1.0], sort=False
                    )
                )
                st.dataframe(scored, use_container_width=True)
                st.download_button(
                    "Download risk scores",
                    scored.to_csv(index=False),
                    file_name="fictional_predictions.csv",
                    mime="text/csv",
                )
            except (ValueError, KeyError) as exc:
                st.error(f"Invalid CSV: {exc}")

with tab_method:
    st.markdown(
        """### What the model does

- Uses 11 UCI predictor columns and excludes potentially ambiguous Status
  and derived Customer Value from the input.
- Chooses one of three algorithms using training-only cross-validation.
- Tunes the decision threshold on separate validation data.
- Provides a descriptive probability estimate for educational exploration.

### What the model does **not** prove

- It has **not** been independently validated on later customers or operators.
- A different, group-disjoint audit still produced a strong ranking ROC-AUC,
  but exact profiles are only proxies for actual customer identity.
- It cannot determine whether offering a discount will prevent any customer
  from leaving or establish net ROI.
- Age fields require fairness and legal review before any real-world use.

Read the methodology and group-robustness audit in the project's GitHub README.
"""
    )
