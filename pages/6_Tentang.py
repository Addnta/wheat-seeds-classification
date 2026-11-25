import streamlit as st

from src.app_data import app_data
from src.data_loader import CLASS_MAP, FEATURES, FEATURE_INFO
from src.ui import apply_style, show_page_intro

st.set_page_config(page_title="Tentang Penelitian", page_icon="📚", layout="wide")
apply_style()
_, data, evaluation = app_data()
show_page_intro("Tentang Penelitian", "Ringkasan sumber data, metode, algoritma, dan batasan penelitian.")

st.markdown("### Judul penelitian")
st.write(
    "Klasifikasi Varietas Biji Gandum Menggunakan Algoritma K-Nearest Neighbor, "
    "Decision Tree, dan Gaussian Naive Bayes."
)
st.markdown("### Tujuan")
st.write(
    "Menerapkan dan membandingkan tiga algoritma klasifikasi pada fitur geometris biji gandum, "
    "serta mengamati pola kesalahan setiap kelas."
)

st.markdown("### Dataset")
st.write(
    f"Dataset yang dimuat berisi {len(data)} sampel, {len(FEATURES)} fitur numerik, dan "
    f"{data['variety'].nunique()} varietas. Distribusi kelas dihitung dari file yang digunakan aplikasi."
)
st.markdown("Sumber resmi: [UCI Machine Learning Repository — Seeds Dataset (ID 236)](https://archive.ics.uci.edu/dataset/236/seeds).")
st.dataframe(
    [{"Label asli": key, "Varietas": value} for key, value in CLASS_MAP.items()],
    width="stretch",
    hide_index=True,
)
st.dataframe(
    [{"Atribut": FEATURE_INFO[key][0], "Keterangan": FEATURE_INFO[key][1]} for key in FEATURES],
    width="stretch",
    hide_index=True,
)

st.markdown("### Algoritma")
st.write("- **K-Nearest Neighbor:** memilih kelas mayoritas di antara tetangga terdekat; lima tetangga dan jarak Euclidean digunakan pada baseline.")
st.write("- **Decision Tree:** membangun aturan percabangan dengan kriteria Gini; konfigurasi baseline tidak membatasi kedalaman dan memakai random state 42.")
st.write("- **Gaussian Naive Bayes:** memperkirakan probabilitas kelas dengan asumsi Gaussian per fitur dan independensi kondisional.")

st.markdown("### Preprocessing dan validasi")
st.write(
    "Label dipisahkan dari tujuh fitur. KNN memakai StandardScaler di dalam Pipeline agar scaler "
    "hanya dipelajari dari bagian training pada fold terkait. Decision Tree dan Gaussian Naive Bayes "
    "menggunakan fitur numerik asli. Evaluasi memakai Stratified 5-Fold Cross-Validation "
    "(shuffle=True, random_state=42)."
)
st.write(
    "Metrik lintas fold adalah accuracy, macro precision, macro recall, dan macro F1-score; "
    "confusion matrix serta classification report berasal dari prediksi out-of-fold."
)
st.caption(f"Evaluasi saat ini membandingkan {len(evaluation.cv_summary)} model dari data aktual.")

st.markdown("### Batasan")
st.write(
    "Hasil hanya menggambarkan dataset benchmark dan konfigurasi baseline ini. Cross-validation "
    "tidak menggantikan validasi eksternal pada sampel, lokasi, musim, atau perangkat pengukuran baru. "
    "Prediksi pada nilai di luar rentang dataset dapat kurang andal."
)
