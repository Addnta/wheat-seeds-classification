import streamlit as st

from src.app_data import app_data, get_final_models
from src.data_loader import CLASS_MAP, FEATURES, FEATURE_INFO
from src.prediction import predict_sample
from src.ui import apply_style, show_page_intro

st.set_page_config(page_title="Prediksi Varietas", page_icon="🌾", layout="wide")
apply_style()
signature, data, evaluation = app_data()
show_page_intro(
    "Prediksi Varietas Biji Gandum",
    "Masukkan tujuh ukuran geometris untuk memperoleh prediksi dari model yang dilatih pada dataset.",
)
st.caption("Model untuk prediksi akhir dilatih pada seluruh data setelah evaluasi CV tersedia.")

model_key = st.selectbox(
    "Algoritma",
    ["knn", "decision_tree", "gaussian_nb"],
    format_func={"knn": "K-Nearest Neighbor", "decision_tree": "Decision Tree", "gaussian_nb": "Gaussian Naive Bayes"}.get,
)
defaults = data.loc[:, FEATURES].median()
values = []
with st.form("prediction_form"):
    st.markdown("### Nilai karakteristik kernel")
    columns = st.columns(2)
    for index, feature in enumerate(FEATURES):
        label, description = FEATURE_INFO[feature]
        with columns[index % 2]:
            values.append(
                st.number_input(
                    label,
                    value=float(defaults[feature]),
                    step=0.001 if feature == "compactness" else 0.01,
                    format="%.4f",
                    help=description,
                    key=f"input_{feature}",
                )
            )
    submitted = st.form_submit_button("Prediksi varietas", type="primary", width="stretch")

st.warning(
    "Prediksi untuk satu atau lebih nilai yang berada di luar rentang fitur pada dataset "
    "perlu ditafsirkan dengan hati-hati dan dapat kurang andal."
)

if submitted:
    try:
        final_models = get_final_models(signature)
        result = predict_sample(final_models, data, model_key, values)
        st.success(f"Prediksi varietas: **{result.class_name}** (label asli {result.class_id}).")
        st.write(f"Model yang digunakan: **{result.model_name}**.")
        st.write(
            "Prediksi dihitung dari nilai masukan menggunakan model yang dilatih pada seluruh "
            "dataset Seeds yang tersedia."
        )
        if result.out_of_range_features:
            names = [FEATURE_INFO[feature][0] for feature in result.out_of_range_features]
            st.warning("Di luar rentang data pada fitur: " + ", ".join(names) + ".")
        if result.probabilities is None:
            st.info("Estimator ini tidak menyediakan probabilitas prediksi.")
        else:
            st.markdown("#### Probabilitas keluaran model")
            probability_table = {
                label: result.probabilities.get(label, 0.0) * 100 for label in CLASS_MAP.values()
            }
            st.bar_chart(probability_table, y_label="Probabilitas (%)", color="#527ca6")
            st.caption("Probabilitas model bukan jaminan kebenaran pada data lapangan.")
    except (ValueError, KeyError, RuntimeError) as exc:
        st.error(str(exc))
