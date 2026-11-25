"""Pelatihan model final dan prediksi sampel yang divalidasi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_loader import CLASS_MAP, FEATURES, LABEL_ORDER
from src.evaluation import EvaluationResults, dataset_fingerprint
from src.models import MODEL_NAMES, build_models
from src.preprocessing import split_features_target


@dataclass
class FinalModelBundle:
    models: dict[str, object]
    dataset_fingerprint: str


@dataclass
class PredictionResult:
    model_key: str
    model_name: str
    class_name: str
    class_id: int
    probabilities: dict[str, float] | None
    out_of_range_features: list[str]


def fit_final_models(
    frame: pd.DataFrame,
    evaluation: EvaluationResults,
    *,
    n_neighbors: int = 5,
    max_depth: int | None = None,
) -> FinalModelBundle:
    """Fit semua model pada seluruh data setelah evaluasi CV terverifikasi."""
    fingerprint = dataset_fingerprint(frame)
    if fingerprint != evaluation.dataset_fingerprint:
        raise ValueError("Dataset berbeda dari dataset yang telah dievaluasi dengan CV.")
    X, y = split_features_target(frame)
    models = build_models(n_neighbors=n_neighbors, max_depth=max_depth)
    for model in models.values():
        model.fit(X, y)
    return FinalModelBundle(models=models, dataset_fingerprint=fingerprint)


def validate_input(values: Sequence[float]) -> np.ndarray:
    """Validasi tujuh nilai terurut, numerik, dan terbatas."""
    if isinstance(values, (str, bytes)) or len(values) != len(FEATURES):
        raise ValueError(f"Masukkan tepat {len(FEATURES)} nilai fitur.")
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Semua nilai fitur harus berupa angka.") from exc
    if numeric.shape != (len(FEATURES),) or not np.isfinite(numeric).all():
        raise ValueError("Nilai fitur harus berupa angka finite (bukan NaN atau tak hingga).")
    return numeric


def predict_sample(
    bundle: FinalModelBundle,
    frame: pd.DataFrame,
    model_key: str,
    values: Sequence[float],
) -> PredictionResult:
    if model_key not in bundle.models:
        raise KeyError(f"Model tidak tersedia: {model_key}")
    if dataset_fingerprint(frame) != bundle.dataset_fingerprint:
        raise ValueError("Dataset prediksi tidak sama dengan dataset pelatihan model.")
    numeric = validate_input(values)
    sample = pd.DataFrame([numeric], columns=FEATURES)
    model = bundle.models[model_key]
    predicted_name = str(model.predict(sample)[0])
    reverse_map = {variety: class_id for class_id, variety in CLASS_MAP.items()}
    if predicted_name not in reverse_map:
        raise RuntimeError(f"Model menghasilkan label yang tidak dikenal: {predicted_name}")

    probabilities: dict[str, float] | None = None
    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(sample)[0]
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probability_values)
        }

    minima = frame.loc[:, FEATURES].min()
    maxima = frame.loc[:, FEATURES].max()
    out_of_range = [
        feature
        for feature, value in zip(FEATURES, numeric)
        if value < minima[feature] or value > maxima[feature]
    ]
    return PredictionResult(
        model_key=model_key,
        model_name=MODEL_NAMES[model_key],
        class_name=predicted_name,
        class_id=reverse_map[predicted_name],
        probabilities=probabilities,
        out_of_range_features=out_of_range,
    )
