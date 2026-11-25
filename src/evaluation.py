"""Cross-validation dan prediksi out-of-fold yang dipisahkan dengan jelas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.data_loader import FEATURES, LABEL_ORDER
from src.models import MODEL_NAMES, build_models
from src.preprocessing import split_features_target

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    X, y = split_features_target(frame)
    values = pd.concat([X, y.rename("variety")], axis=1)
    return str(int(pd.util.hash_pandas_object(values, index=True).sum()))


@dataclass
class EvaluationResults:
    """Metrik mean ± SD per fold dan keluaran gabungan OOF."""

    cv_summary: pd.DataFrame
    cv_fold_scores: dict[str, dict[str, np.ndarray]]
    oof_predictions: dict[str, pd.Series]
    oof_metrics: dict[str, dict[str, float]]
    confusion_matrices: dict[str, pd.DataFrame]
    classification_reports: dict[str, pd.DataFrame]
    dataset_fingerprint: str


def evaluate_models(
    frame: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    max_depth: int | None = None,
) -> EvaluationResults:
    X, y = split_features_target(frame)
    if len(frame) < N_SPLITS:
        raise ValueError(f"Minimal {N_SPLITS} baris diperlukan untuk {N_SPLITS}-fold CV.")
    class_counts = y.value_counts()
    if class_counts.min() < N_SPLITS:
        raise ValueError("Setiap kelas harus memiliki minimal 5 sampel untuk Stratified 5-Fold CV.")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(n_neighbors=n_neighbors, max_depth=max_depth)
    rows: list[dict[str, float | str]] = []
    fold_scores: dict[str, dict[str, np.ndarray]] = {}
    oof_predictions: dict[str, pd.Series] = {}
    oof_metrics: dict[str, dict[str, float]] = {}
    confusion_matrices: dict[str, pd.DataFrame] = {}
    classification_reports: dict[str, pd.DataFrame] = {}

    for key, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1,
        )
        row: dict[str, float | str] = {"model_key": key, "model": MODEL_NAMES[key]}
        fold_scores[key] = {}
        for metric in SCORING:
            values = np.asarray(scores[f"test_{metric}"], dtype=float)
            fold_scores[key][metric] = values
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)

        predicted = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        prediction_series = pd.Series(predicted, index=frame.index, name="predicted_variety")
        oof_predictions[key] = prediction_series
        report_dict = classification_report(
            y,
            predicted,
            labels=LABEL_ORDER,
            target_names=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        oof_metrics[key] = {
            "accuracy": float(report_dict["accuracy"]),
            "precision_macro": float(report_dict["macro avg"]["precision"]),
            "recall_macro": float(report_dict["macro avg"]["recall"]),
            "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        }
        confusion_matrices[key] = pd.DataFrame(
            confusion_matrix(y, predicted, labels=LABEL_ORDER),
            index=pd.Index(LABEL_ORDER, name="Aktual"),
            columns=pd.Index(LABEL_ORDER, name="Prediksi"),
        )
        classification_reports[key] = pd.DataFrame(report_dict).T

    cv_summary = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    return EvaluationResults(
        cv_summary=cv_summary,
        cv_fold_scores=fold_scores,
        oof_predictions=oof_predictions,
        oof_metrics=oof_metrics,
        confusion_matrices=confusion_matrices,
        classification_reports=classification_reports,
        dataset_fingerprint=dataset_fingerprint(frame),
    )


def best_model(results: EvaluationResults, metric: str = "f1_macro") -> str:
    if metric not in {"accuracy", "f1_macro"}:
        raise ValueError("Metrik utama harus accuracy atau f1_macro.")
    column = f"{metric}_mean"
    winner = results.cv_summary.loc[results.cv_summary[column].idxmax(), "model_key"]
    return str(winner)
