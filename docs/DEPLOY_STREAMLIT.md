# Public Streamlit demonstration — deployment steps

## Public application

**Live URL:** [https://andres-churn-prediction.streamlit.app/](https://andres-churn-prediction.streamlit.app/)

The owner has deployed the app in Streamlit Community Cloud. The initialization
flow and a single-customer prediction have been visually verified on the live
instance (the example returned a 1.3% risk, below the validation alert threshold
of 0.27). The batch CSV and hypothetical business scenario still require
end-to-end interactive verification.

The steps below describe how to deploy again, if hosting needs to be reset.

## Publish on Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **New app** → **Deploy a public app from GitHub**.
3. Repository: `andres-olivera/customer-churn-prediction`.
4. Branch: `main`.
5. Main file path: `app/streamlit_app.py`.
6. Python version: 3.11 or 3.12, if the host asks.
7. Deploy. The root `requirements.txt` installs `-e .[app]`.
8. Click **Initialize ML demonstration** when the new application loads.
   On first boot, the server fetches original data from UCI, validates rows,
   and trains the calibrated model. This may take several minutes.
9. After successful loading, try the fictional CSV download/upload and
   share the Streamlit URL on the README and in your applications.

## Data and hosting limitations

- The dataset comes from https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset.
- The first run requires server-side outbound internet access to UCI.
- Instances may sleep or reset. On a cold restart you may need to initialize again.
- No real customer information should be uploaded to this demonstration.
- Data, model files and generated reports stored on the Cloud instance are
  ephemeral. GitHub stores the canonical reproducible reports instead.
- A public demo is **not** proof of deployment readiness for operational use.

If deployment fails due to UCI rate limits, initialize locally or rerun later.
Avoid committing executable pickles from untrusted sources.
