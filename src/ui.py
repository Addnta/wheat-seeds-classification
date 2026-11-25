"""Komponen tampilan bersama untuk seluruh halaman Streamlit."""

import streamlit as st

from src.data_loader import FEATURE_INFO


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #ffffff; color: #182230; }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.94); }
        [data-testid="stSidebar"] { background: #f7f9fc; border-right: 1px solid #e7ebf0; }
        h1, h2, h3 { color: #172b4d; letter-spacing: -0.02em; }
        p, label, li { color: #344054; }
        [data-testid="stMetric"] {
            border: 1px solid #e5eaf0; border-radius: 12px; padding: 16px;
            background: #ffffff;
        }
        [data-testid="stMetricValue"] { color: #183b66; }
        div[data-testid="stDataFrame"] { border: 1px solid #e5eaf0; border-radius: 8px; }
        .muted { color: #667085; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_name(feature: str) -> str:
    return FEATURE_INFO.get(feature, (feature.replace("_", " ").title(), ""))[0]


def show_page_intro(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
