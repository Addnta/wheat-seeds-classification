"""Cache Streamlit untuk data, evaluasi, dan model prediksi final."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import DATA_PATH, FALLBACK_PATH, load_dataset
from src.evaluation import EvaluationResults, evaluate_models
from src.prediction import FinalModelBundle, fit_final_models


def _dataset_signature() -> str:
    path = DATA_PATH if DATA_PATH.is_file() else FALLBACK_PATH
    if not path.is_file():
        return "missing"
    stat = path.stat()
    return f"{Path(path).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_data(show_spinner="Memuat dataset Seeds UCI...")
def get_dataset(signature: str):
    del signature  # Argumen ini menjadi kunci cache saat isi file berubah.
    return load_dataset()


@st.cache_data(show_spinner="Menghitung evaluasi Stratified 5-Fold...")
def get_evaluation(signature: str, n_neighbors: int = 5, max_depth: int | None = None) -> EvaluationResults:
    frame = get_dataset(signature)
    return evaluate_models(frame, n_neighbors=n_neighbors, max_depth=max_depth)


@st.cache_resource(show_spinner="Melatih model akhir dengan seluruh dataset...")
def get_final_models(
    signature: str, n_neighbors: int = 5, max_depth: int | None = None
) -> FinalModelBundle:
    frame = get_dataset(signature)
    evaluation = get_evaluation(signature, n_neighbors, max_depth)
    return fit_final_models(
        frame,
        evaluation,
        n_neighbors=n_neighbors,
        max_depth=max_depth,
    )


def app_data():
    signature = _dataset_signature()
    frame = get_dataset(signature)
    evaluation = get_evaluation(signature)
    return signature, frame, evaluation
