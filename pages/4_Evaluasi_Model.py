import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.app_data import app_data
from src.data_loader import LABEL_ORDER
from src.models import MODEL_NAMES
from src.ui import apply_style, show_page_intro

st.set_page_config(page_title="Evaluasi Model", page_icon="🧪", layout="wide")
apply_style()
_, _, evaluation = app_data()
show_page_intro("Evaluasi Model", "Hasil rata-rata lintas fold dan agregasi prediksi OOF ditampilkan terpisah.")

st.markdown("### Hasil Stratified 5-Fold Cross-Validation")
st.caption("Setiap rata-rata dan simpangan baku dihitung dari lima fold, terpisah dari prediksi OOF gabungan.")
primary_metric = st.selectbox(
    "Metrik utama pemilihan model",
    options=["accuracy", "f1_macro"],
    format_func={"accuracy": "Akurasi", "f1_macro": "Macro F1-Score"}.get,
)
metric_labels = {
    "accuracy": "Akurasi",
    "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall",
    "f1_macro": "Macro F1-Score",
}
mean_column = f"{primary_metric}_mean"
winner = evaluation.cv_summary.loc[evaluation.cv_summary[mean_column].idxmax()]
st.success(
    f"Model teratas untuk {metric_labels[primary_metric]} adalah "
    f"{winner['model']} (rata-rata CV = {winner[mean_column]:.4f})."
)

table_columns = ["model"]
for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
    table_columns.extend([f"{metric}_mean", f"{metric}_std"])
display = evaluation.cv_summary[table_columns].rename(
    columns={
        "model": "Model",
        "accuracy_mean": "Rata-rata akurasi CV",
        "accuracy_std": "SD akurasi CV",
        "precision_macro_mean": "Rata-rata Macro Precision CV",
        "precision_macro_std": "SD Macro Precision CV",
        "recall_macro_mean": "Rata-rata Macro Recall CV",
        "recall_macro_std": "SD Macro Recall CV",
        "f1_macro_mean": "Rata-rata Macro F1 CV",
        "f1_macro_std": "SD Macro F1 CV",
    }
)
st.dataframe(
    display.style.format({column: "{:.4f}" for column in display.columns if column != "Model"}),
    width="stretch",
    hide_index=True,
)

chart = evaluation.cv_summary.set_index("model")[["accuracy_mean", "f1_macro_mean"]]
chart.columns = ["Rata-rata akurasi CV", "Rata-rata Macro F1 CV"]
st.bar_chart(chart, y_label="Rata-rata skor lima fold", color=["#315b86", "#87a9ca"])

st.markdown("### Rincian model")
selected_key = st.selectbox(
    "Pilih model untuk melihat matriks konfusi dan laporan klasifikasi",
    list(MODEL_NAMES),
    format_func=MODEL_NAMES.get,
)
left, right = st.columns([1, 1.25])
with left:
    st.markdown("#### Matriks konfusi OOF")
    matrix = evaluation.confusion_matrices[selected_key]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Prediksi", ylabel="Aktual")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
with right:
    st.markdown("#### Laporan klasifikasi OOF")
    report = evaluation.classification_reports[selected_key].copy()
    st.dataframe(
        report.style.format({c: "{:.4f}" for c in report.columns if c != "support"}),
        width="stretch",
    )

with st.expander("Perbedaan hasil CV dan OOF"):
    st.write(
        "**Rata-rata CV** merata-ratakan metrik yang dihitung terpisah pada setiap fold. "
        "**OOF** menggabungkan satu prediksi untuk setiap sampel dari model yang tidak "
        "melatih sampel tersebut, lalu menghitung metrik gabungan. Karena agregasinya "
        "berbeda, nilai Macro Precision/Recall/F1 OOF dapat sedikit berbeda dari rata-rata CV."
    )
    oof_metric = evaluation.oof_metrics[selected_key]
    st.dataframe(
        [{"Metrik agregasi OOF": metric_labels[key], "Nilai OOF": value} for key, value in oof_metric.items()],
        width="stretch",
        hide_index=True,
    )

st.markdown("### Cara kerja dan batasan model")
explanations = {
    "knn": (
        "KNN memberi label berdasarkan mayoritas dari lima tetangga terdekat dengan jarak Euclidean. "
        "Fitur distandardisasi di dalam Pipeline pada setiap fold. Hasil sensitif terhadap skala, "
        "nilai k, dan pilihan jarak."
    ),
    "decision_tree": (
        "Decision Tree memecah data melalui aturan fitur yang menurunkan impurity Gini. "
        "Pohon mudah ditelusuri, tetapi pohon tanpa batas kedalaman dapat sensitif terhadap "
        "perubahan data dan berisiko overfitting."
    ),
    "gaussian_nb": (
        "Gaussian Naive Bayes menghitung probabilitas kelas dengan asumsi fitur berdistribusi "
        "normal dan independen secara kondisional. Prosesnya ringan, tetapi korelasi antarfitur "
        "dapat membatasi kecocokan asumsi."
    ),
}
for key, name in MODEL_NAMES.items():
    st.markdown(f"**{name}.** {explanations[key]}")
st.caption("Semua hasil terbatas pada dataset Seeds UCI dan konfigurasi baseline yang dievaluasi.")
