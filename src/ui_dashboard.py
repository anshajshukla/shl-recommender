"""
SHL Assessment Recommendation System

Professional assessment matching platform for talent acquisition.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
)

st.title("📊 SHL Recommender Dashboard")


def call_health_check(base_url: str) -> Optional[Dict]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        if response.ok:
            return response.json()
    except requests.RequestException:
        return None
    return None


def request_recommendations(
    base_url: str,
    query: str,
    top_k: int,
    custom_ratio: Optional[Dict[str, float]] = None,
) -> Dict:
    payload: Dict[str, object] = {"query": query, "top_k": top_k}
    if custom_ratio:
        payload["test_type_ratio"] = custom_ratio

    response = requests.post(
        f"{base_url.rstrip('/')}/recommend",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Configuration")
    api_base_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE_URL,
        help="The URL where the FastAPI service is running.",
    )
    top_k = st.slider("Results (top_k)", min_value=1, max_value=20, value=10)

    enable_custom_ratio = st.checkbox("Override K/P ratio")
    ratio = None
    if enable_custom_ratio:
        k_ratio = st.number_input(
            "K (Knowledge) ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Fraction of Knowledge/Technical assessments.",
        )
        p_ratio = 1.0 - k_ratio
        st.markdown(f"P (Personality) ratio: **{p_ratio:.2f}**")
        ratio = {"K": round(k_ratio, 2), "P": round(p_ratio, 2)}

    st.markdown("---")
    if st.button("Check API health"):
        health = call_health_check(api_base_url)
        if health:
            st.success(f"API is healthy. Uptime: {health.get('uptime_seconds', 0):.0f}s")
            st.json(health)
        else:
            st.error("Failed to connect to the API health endpoint.")


query_input = st.text_area(
    "Enter job query or description",
    placeholder="Example: Hiring a finance analyst with strong Excel and budgeting skills.",
    height=140,
)

submit = st.button("Get Recommendations", type="primary", use_container_width=True)

if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

if submit:
    if not query_input.strip():
        st.warning("Please provide a query before requesting recommendations.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                result = request_recommendations(api_base_url, query_input, top_k, ratio)
                st.session_state["last_response"] = result
            except requests.HTTPError as http_err:
                st.session_state["last_response"] = None
                st.error(f"API error: {http_err.response.text}")
            except requests.RequestException as req_err:
                st.session_state["last_response"] = None
                st.error(f"Request failed: {req_err}")
            except Exception as unexpected:
                st.session_state["last_response"] = None
                st.error(f"Unexpected error: {unexpected}")


raw = st.session_state.get("last_response")
recommendations: List[Dict] = (raw or {}).get("recommended_assessments", [])

if recommendations:
    # KPIs
    durations = []
    for rec in recommendations:
        duration_val = rec.get("duration", rec.get("duration_minutes"))
        if isinstance(duration_val, (int, float)):
            durations.append(int(duration_val))
        elif isinstance(duration_val, str) and duration_val.isdigit():
            durations.append(int(duration_val))
    
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Results", len(recommendations))
    col2.metric("Avg Duration (min)", avg_duration)
    col3.metric("K/P Override", "Yes" if ratio else "No")

    # Table
    table_rows = []
    for rec in recommendations:
        test_type = ", ".join(rec.get("test_type", []))
        table_rows.append(
            {
                "Name": rec.get("name", ""),
                "URL": rec.get("url", ""),
                "Test Type": test_type,
                "Duration (min)": rec.get("duration", rec.get("duration_minutes", "")),
                "Adaptive": rec.get("adaptive_support", ""),
                "Remote": rec.get("remote_support", ""),
                "Description": rec.get("description", ""),
            }
        )
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Charts
    c1, c2 = st.columns(2)
    try:
        # Test type pie (flatten list)
        test_types_all: List[str] = []
        for rec in recommendations:
            for t in rec.get("test_type", []):
                test_types_all.append(str(t))
        if test_types_all:
            pie_df = pd.Series(test_types_all, name="Test Type").value_counts().reset_index()
            pie_df.columns = ["Test Type", "Count"]
            fig_pie = px.pie(pie_df, names="Test Type", values="Count", title="Test Type Distribution")
            c1.plotly_chart(fig_pie, use_container_width=True)
    except Exception as e:
        c1.warning(f"Could not generate test type chart: {str(e)}")

    try:
        if durations:
            fig_hist = px.histogram(x=durations, nbins=10, title="Duration Distribution (min)")
            fig_hist.update_layout(xaxis_title="Minutes", yaxis_title="Count")
            c2.plotly_chart(fig_hist, use_container_width=True)
    except Exception as e:
        c2.warning(f"Could not generate duration chart: {str(e)}")

with st.expander("Show raw response", expanded=False):
    if raw:
        st.code(json.dumps(raw, indent=2), language="json")
    else:
        st.write("No response available yet. Submit a query to view details.")

