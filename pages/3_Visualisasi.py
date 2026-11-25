import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.app_data import app_data
from src.data_loader import FEATURES, LABEL_ORDER
from src.ui import apply_style, feature_name, show_page_intro

st.set_page_config(page_title="Visualisasi Data", page_icon="📊", layout="wide")
apply_style()
_, data, _ = app_data()
show_page_intro("Visualisasi Data", "Grafik eksplorasi dihitung langsung dari nilai dataset.")
sns.set_theme(style="whitegrid", palette=["#315b86", "#8aa7c7", "#c49a6c"])

st.markdown("### Distribusi varietas")
counts = data["variety"].value_counts().reindex(LABEL_ORDER)
st.bar_chart(counts, color="#315b86")
st.caption("Distribusi menunjukkan jumlah sampel aktual pada setiap label varietas.")

left, right = st.columns(2)
with left:
    st.markdown("### Distribusi luas kernel per varietas")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=data, x="variety", y="area", order=LABEL_ORDER, ax=ax)
    ax.set(xlabel="Varietas", ylabel="Luas kernel", title="Sebaran luas per kelas")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Median dan rentang antarkuartil memperlihatkan variasi luas pada tiap varietas.")
with right:
    st.markdown("### Panjang dan lebar kernel")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=data,
        x="kernel_length",
        y="kernel_width",
        hue="variety",
        hue_order=LABEL_ORDER,
        s=48,
        alpha=0.8,
        ax=ax,
    )
    ax.set(xlabel="Panjang kernel", ylabel="Lebar kernel", title="Panjang vs lebar")
    ax.legend(title="Varietas", frameon=True)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Titik yang berdekatan menunjukkan sampel dengan ukuran panjang dan lebar serupa.")

st.markdown("### Distribusi kekompakan")
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(data=data, x="compactness", hue="variety", hue_order=LABEL_ORDER, bins=18, alpha=0.45, ax=ax)
ax.set(xlabel="Kekompakan (compactness)", ylabel="Jumlah sampel", title="Distribusi kekompakan per varietas")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

left, right = st.columns(2)
with left:
    st.markdown("### Korelasi fitur")
    correlation = data.loc[:, FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        xticklabels=[feature_name(x) for x in FEATURES],
        yticklabels=[feature_name(x) for x in FEATURES],
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Korelasi menggambarkan hubungan linear pada dataset, bukan sebab-akibat.")
with right:
    st.markdown("### Boxplot atribut pilihan")
    selected_features = st.multiselect(
        "Pilih fitur",
        FEATURES,
        default=["area", "compactness", "kernel_length", "kernel_width"],
        format_func=feature_name,
    )
    if selected_features:
        melted = data.melt(
            id_vars="variety", value_vars=selected_features,
            var_name="feature", value_name="nilai",
        )
        melted["feature_label"] = melted["feature"].map(feature_name)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=melted, x="feature_label", y="nilai", hue="variety", hue_order=LABEL_ORDER, ax=ax)
        ax.set(xlabel="Fitur", ylabel="Nilai", title="Sebaran fitur yang dipilih")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Varietas")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Pilih setidaknya satu fitur untuk menampilkan boxplot.")
