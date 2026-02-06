from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("API_URL", "http://localhost:8000")
API_URL_DEFAULT = f"{API_BASE}/evaluate-csv"

st.set_page_config(
    page_title="CoDiet Nutrition Evaluator",
    layout="centered",
    page_icon="🥗",
    initial_sidebar_state="collapsed",
)

css_file = Path(__file__).parent / "styles/styles.css"
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-header">
        <h1>🥗 CoDiet Nutrition Evaluator</h1>
        <p>Result testing and visualization for CoDiet nutrition models</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown("### About")
st.markdown(
    "Upload a CoDiet-style CSV file. The app sends it to the backend and returns model prediction summaries "
    "and optional regression metrics when a numeric target column is present."
)
st.markdown("---")

api_url = st.text_input("FastAPI Endpoint URL", value=API_URL_DEFAULT)
uploaded = st.file_uploader("Choose a CSV file to evaluate", type=["csv"])
run_eval = st.button("Run Evaluation", disabled=uploaded is None, type="primary")

if run_eval:
    if uploaded is None:
        st.error("Please upload a CSV file.")
        st.stop()

    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        with st.spinner("Uploading and evaluating..."):
            r = requests.post(api_url, files=files, timeout=120)
    except requests.RequestException as e:
        st.error(f"Failed to reach API: {e}")
        st.info("Make sure the backend is running: `uv run invoke serve-api`")
        st.stop()

    if r.status_code != 200:
        st.error(f"API error {r.status_code}")
        try:
            st.json(r.json())
        except Exception:
            st.code(r.text)
        st.stop()

    result = r.json()
    st.success("Evaluation complete")
    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", result.get("n_samples"))
    c2.metric("Features", result.get("n_features"))
    c3.metric("Pred Mean", f"{result.get('prediction_mean', float('nan')):.3f}")

    if result.get("has_labels"):
        c4, c5 = st.columns(2)
        c4.metric("MAE", f"{result.get('mae', float('nan')):.4f}")
        c5.metric("RMSE", f"{result.get('rmse', float('nan')):.4f}")

    with st.expander("Full Response (JSON)"):
        st.json(result)
