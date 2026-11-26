"""Entry point dan navigasi enam halaman aplikasi Streamlit."""

import streamlit as st

st.set_page_config(
    page_title="Wheat Seeds Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

page_specs = [
    ("pages/1_Dashboard.py", "Dashboard", "📊"),
    ("pages/2_Dataset.py", "Dataset", "📋"),
    ("pages/3_Visualisasi.py", "Visualisasi", "📈"),
    ("pages/4_Evaluasi_Model.py", "Evaluasi Model", "🧪"),
    ("pages/5_Prediksi.py", "Prediksi", "🌾"),
    ("pages/6_Tentang.py", "Tentang Penelitian", "📚"),
]
pages = [
    st.Page(path, title=title, icon=icon, default=(index == 0))
    for index, (path, title, icon) in enumerate(page_specs)
]

if "dark_mode_preference" not in st.session_state:
    st.session_state.dark_mode_preference = False


def toggle_theme() -> None:
    st.session_state.dark_mode_preference = not st.session_state.dark_mode_preference


with st.container(key="glass_navigation"):
    nav_columns = st.columns([1, 1, 1, 1.35, 1, 1.55, 1.15], vertical_alignment="center")
    for column, page, (_, title, icon) in zip(nav_columns[:6], pages, page_specs):
        with column:
            st.page_link(page, label=title, icon=icon)
    with nav_columns[6]:
        dark_mode = st.session_state.dark_mode_preference
        st.button(
            "☀️" if dark_mode else "🌙",
            key="theme_button",
            on_click=toggle_theme,
            help="Ganti ke mode terang" if dark_mode else "Ganti ke mode gelap",
        )

st.navigation({"Analisis": pages}, position="hidden").run()
