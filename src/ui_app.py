"""
Streamlit frontend for the SHL Assessment Recommender.

Features:
- Text area for entering a hiring query
- Optional custom top_k selection
- Optional K/P ratio override
- Displays recommendations returned by the FastAPI backend
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="wide",
)

st.title("SHL Assessment Recommender")

if "last_response" not in st.session_state:
    st.session_state["last_response"] = None


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
    payload: Dict[str, object] = {
        "query": query,
        "top_k": top_k,
    }
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
    height=200,
)

submit = st.button("Get Recommendations", type="primary")

if submit:
    if not query_input.strip():
        st.warning("Please provide a query before requesting recommendations.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                result = request_recommendations(api_base_url, query_input, top_k, ratio)
                st.session_state["last_response"] = result
                recommendations: List[Dict] = result.get("recommended_assessments", [])
                if recommendations:
                    st.success(f"Received {len(recommendations)} recommendations.")
                    df = []
                    for rec in recommendations:
                        df.append(
                            {
                                "Name": rec.get("name", ""),
                                "Duration (min)": rec.get("duration", ""),
                                "Test Type": ", ".join(rec.get("test_type", [])),
                                "Adaptive": rec.get("adaptive_support", ""),
                                "Remote": rec.get("remote_support", ""),
                                "URL": rec.get("url", ""),
                                "Description": rec.get("description", ""),
                            }
                        )
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No recommendations returned for this query.")
            except requests.HTTPError as http_err:
                st.error(f"API error: {http_err.response.text}")
                st.session_state["last_response"] = None
            except requests.RequestException as req_err:
                st.error(f"Request failed: {req_err}")
                st.session_state["last_response"] = None
            except Exception as unexpected:
                st.error(f"Unexpected error: {unexpected}")
                st.session_state["last_response"] = None


with st.expander("Show raw response", expanded=False):
    raw = st.session_state.get("last_response")
    if raw:
        st.code(json.dumps(raw, indent=2), language="json")
    else:
        st.write("No response available yet. Submit a query to view details.")

