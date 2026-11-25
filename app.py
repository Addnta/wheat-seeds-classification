"""Entry point dan navigasi enam halaman aplikasi Streamlit."""

import streamlit as st

st.set_page_config(
    page_title="Wheat Seeds Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Analisis": [
        st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("pages/2_Dataset.py", title="Dataset", icon="📋"),
        st.Page("pages/3_Visualisasi.py", title="Visualisasi", icon="📈"),
        st.Page("pages/4_Evaluasi_Model.py", title="Evaluasi Model", icon="🧪"),
        st.Page("pages/5_Prediksi.py", title="Prediksi", icon="🌾"),
        st.Page("pages/6_Tentang.py", title="Tentang Penelitian", icon="📚"),
    ]
}
st.navigation(pages).run()
