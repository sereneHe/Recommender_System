from __future__ import annotations
import os
from pathlib import Path

import requests
import streamlit as st

# Use the environment variable set in docker-compose
API_BASE = os.environ.get("API_URL", "http://localhost:8000")
API_URL_DEFAULT = f"{API_BASE}/evaluate-csv"  # "http://127.0.0.1:8000/evaluate-csv"

# Page configuration with custom theme
st.set_page_config(
    page_title="Breast Cancer Evaluator", layout="centered", page_icon="🩺", initial_sidebar_state="collapsed"
)

# Load custom CSS from external file
css_file = Path(__file__).parent / "styles/styles.css"
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🩺 Breast Cancer Evaluator</h1>
        <p>MLOps Group 5 • 2026</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Introduction section
with st.container():
    st.markdown("### 📊 About This Application")
    st.markdown(
        """
        Upload a CSV dataset. The app sends it to the FastAPI backend for preprocessing + evaluation using a pre-trained model.
    """
    )

    with st.expander("ℹ️ Dataset Information"):
        st.markdown(
            """
            **Wisconsin Breast Cancer Dataset**

            The model is trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

            🔗 [Download dataset from Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
        """
        )

st.markdown("---")

# Configuration section
st.markdown("### ⚙️ Configuration")
col_config1, col_config2 = st.columns([1, 1])

with col_config1:
    api_url = st.text_input(
        "FastAPI Endpoint URL", value=API_URL_DEFAULT, help="The backend API endpoint for model evaluation"
    )

with col_config2:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛠️ Backend Setup"):
        st.code("uv run invoke serve-api", language="bash")
        st.caption("Start the backend before evaluating")

st.markdown("---")

# Upload section
st.markdown("### 📁 Upload Dataset")
uploaded = st.file_uploader(
    "Choose a CSV file to evaluate", type=["csv"], help="Upload your breast cancer dataset in CSV format"
)

if uploaded:
    st.success(f"✅ File uploaded: **{uploaded.name}**")

st.markdown("---")

# Evaluate button
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
with col_btn2:
    run_eval = st.button("🚀 Evaluate Dataset", disabled=uploaded is None, type="primary")

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
        st.info("💡 Make sure the backend is running: `uv run invoke serve-api`")
        st.stop()

    if r.status_code != 200:
        st.error(f"API error {r.status_code}")
        try:
            st.json(r.json())
        except Exception:
            st.code(r.text)
        st.stop()

    result = r.json()

    # Success message
    st.success("✅ Evaluation Complete!")

    st.markdown("---")
    st.markdown("### 📈 Results")

    # Display key metrics with enhanced styling
    if result.get("has_labels"):
        # Metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="🎯 Accuracy", value=f"{result['accuracy']*100:.2f}%", delta=None)

        with col2:
            st.metric(label="✅ Correct Predictions", value=result["correct"])

        with col3:
            st.metric(label="📊 Total Samples", value=result["total"])

        # Progress bar for accuracy
        st.markdown("#### Performance Visualization")
        st.progress(result["accuracy"])

        # Additional details in expandable section
        with st.expander("📋 Detailed Results (JSON)"):
            st.json(result)

    else:
        st.info("ℹ️ " + result.get("message", "No labels found in dataset."))
        if result.get("predicted_positive") is not None:
            st.metric(label="Predicted Positive Cases", value=result.get("predicted_positive"))

        with st.expander("📋 Full Response (JSON)"):
            st.json(result)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Part of MLOps DTU Project. </p>
        <p style='font-size: 0.9rem;'><a href="https://github.com/kadijairus/mlo_project" target="_blank">Github project</a> • 2026</p>
    </div>
""",
    unsafe_allow_html=True,
)
