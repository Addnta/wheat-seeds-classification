import streamlit as st

from src.app_data import app_data
from src.data_loader import FEATURES, LABEL_ORDER
from src.evaluation import best_model
from src.models import MODEL_NAMES
from src.ui import apply_style, show_page_intro

apply_style()
_, data, evaluation = app_data()
show_page_intro(
    "Wheat Seeds Classification",
    "Analisis dan Klasifikasi Varietas Biji Gandum Menggunakan Machine Learning",
)
metric_key = st.selectbox(
    "Metrik utama untuk menentukan model terbaik",
    options=["f1_macro", "accuracy"],
    format_func={"f1_macro": "Macro F1-Score", "accuracy": "Akurasi"}.get,
)
winner_key = best_model(evaluation, metric_key)
winner = evaluation.cv_summary.set_index("model_key").loc[winner_key]
metric_label = {"f1_macro": "Macro F1-Score", "accuracy": "Akurasi"}[metric_key]

cols = st.columns(4)
cols[0].metric("Jumlah sampel", f"{len(data):,}")
cols[1].metric("Fitur numerik", str(len(FEATURES)))
cols[2].metric("Varietas", str(data["variety"].nunique()))
cols[3].metric("Model terbaik", MODEL_NAMES[winner_key])

st.markdown("### Ringkasan penelitian")
st.write(
    "Penelitian membandingkan K-Nearest Neighbor, Decision Tree, dan Gaussian Naive Bayes "
    "untuk mengklasifikasikan varietas gandum dari karakteristik geometris kernel. "
    "Evaluasi memakai Stratified 5-Fold Cross-Validation dengan pembagian yang dapat direproduksi."
)
left, right = st.columns(2)
with left:
    st.markdown("### Distribusi kelas")
    distribution = data["variety"].value_counts().reindex(LABEL_ORDER).rename("Jumlah sampel")
    st.bar_chart(distribution, color="#315b86")
with right:
    st.markdown("### Perbandingan kinerja")
    chart = evaluation.cv_summary.set_index("model")[["accuracy_mean", "f1_macro_mean"]]
    chart.columns = ["Rata-rata akurasi CV", "Rata-rata Macro F1 CV"]
    st.bar_chart(chart, y_label="Skor rata-rata CV", color=["#315b86", "#87a9ca"])

st.markdown("### Ringkasan model teratas")
st.write(
    f"**{MODEL_NAMES[winner_key]}** memperoleh rata-rata "
    f"{metric_label} sebesar {winner[f'{metric_key}_mean']:.4f} "
    f"(simpangan baku {winner[f'{metric_key}_std']:.4f}) pada lima fold. "
    "Kesimpulan ini hanya berlaku untuk dataset dan konfigurasi baseline yang diuji."
)
st.dataframe(
    evaluation.cv_summary[["model", "accuracy_mean", "f1_macro_mean"]].rename(
        columns={"accuracy_mean": "Rata-rata akurasi CV", "f1_macro_mean": "Rata-rata Macro F1 CV"}
    ).style.format({"Rata-rata akurasi CV": "{:.4f}", "Rata-rata Macro F1 CV": "{:.4f}"}),
    width="stretch",
    hide_index=True,
)
class_counts = data["class_id"].value_counts().to_dict()
if len(data) != 210 or class_counts != {1: 70, 2: 70, 3: 70}:
    st.warning(
        "Dataset yang dimuat berbeda dari spesifikasi Seeds UCI yang diharapkan "
        "(210 sampel, masing-masing 70 per kelas). Data tidak diubah; tinjau halaman Dataset."
    )
