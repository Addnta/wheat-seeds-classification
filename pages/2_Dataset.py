import streamlit as st

from src.app_data import app_data
from src.data_loader import FEATURES, FEATURE_INFO, LABEL_ORDER, TARGET_COLUMN, VARIETY_COLUMN, dataset_diagnostics
from src.ui import apply_style, feature_name, show_page_intro

st.set_page_config(page_title="Eksplorasi Dataset", page_icon="📋", layout="wide")
apply_style()
_, data, _ = app_data()
diagnostics = dataset_diagnostics(data)

show_page_intro("Eksplorasi Dataset", "Pemeriksaan kualitas dan isi dataset Seeds UCI.")
col1, col2, col3 = st.columns(3)
col1.metric("Observasi", f"{diagnostics['rows']:,}")
col2.metric("Fitur", str(diagnostics["feature_count"]))
col3.metric("Label kelas", str(data[VARIETY_COLUMN].nunique()))

st.markdown("### Tabel data")
selected_classes = st.multiselect("Filter varietas", LABEL_ORDER, default=list(LABEL_ORDER))
max_rows = max(1, len(data))
row_limit = st.slider("Jumlah baris yang ditampilkan", 5, max_rows, min(30, max_rows), step=5)
filtered = data[data[VARIETY_COLUMN].isin(selected_classes)]
display_columns = [*FEATURES, TARGET_COLUMN, VARIETY_COLUMN]
st.dataframe(filtered.loc[:, display_columns].head(row_limit), width="stretch", hide_index=True)
st.caption(f"Menampilkan {min(row_limit, len(filtered))} dari {len(filtered)} baris sesuai filter.")

st.markdown("### Pemeriksaan kualitas")
left, right = st.columns(2)
with left:
    st.write("**Tipe data**")
    st.dataframe(diagnostics["dtypes"].rename("Tipe").to_frame(), width="stretch")
    st.write("**Nilai kosong per kolom**")
    st.dataframe(diagnostics["missing_values"].rename("Jumlah kosong").to_frame(), width="stretch")
with right:
    st.metric("Duplikasi fitur + label", diagnostics["duplicate_rows"])
    st.caption("Baris duplikat hanya dilaporkan dan tidak dihapus otomatis.")
    st.write("**Distribusi kelas**")
    distribution = diagnostics["class_distribution"].rename_axis("Varietas").rename("Jumlah")
    st.dataframe(distribution.to_frame(), width="stretch")
    st.write("**Statistik deskriptif fitur**")
    st.dataframe(diagnostics["feature_statistics"].rename(index=feature_name), width="stretch")

st.markdown("### Arti atribut")
attribute_table = [
    {"Atribut": FEATURE_INFO[key][0], "Keterangan": FEATURE_INFO[key][1]}
    for key in FEATURES
]
st.dataframe(attribute_table, width="stretch", hide_index=True)
