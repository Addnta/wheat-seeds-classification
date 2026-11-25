"""Definisi baseline KNN, Decision Tree, dan Gaussian Naive Bayes."""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DEFAULT_K = 5

MODEL_NAMES = {
    "knn": "K-Nearest Neighbor",
    "decision_tree": "Decision Tree",
    "gaussian_nb": "Gaussian Naive Bayes",
}


def build_model(model_key: str, *, n_neighbors: int = DEFAULT_K, max_depth: int | None = None):
    """Bangun estimator sesuai baseline notebook dengan opsi parameter terbatas."""
    if model_key == "knn":
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("Jumlah tetangga KNN harus berupa bilangan bulat positif.")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights="uniform",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        )
    if model_key == "decision_tree":
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("Kedalaman Decision Tree harus None atau bilangan bulat positif.")
        return DecisionTreeClassifier(
            criterion="gini", max_depth=max_depth, random_state=RANDOM_STATE
        )
    if model_key == "gaussian_nb":
        return GaussianNB(var_smoothing=1e-9)
    raise KeyError(f"Model tidak dikenal: {model_key}")


def build_models(*, n_neighbors: int = DEFAULT_K, max_depth: int | None = None):
    return {
        key: build_model(key, n_neighbors=n_neighbors, max_depth=max_depth)
        for key in MODEL_NAMES
    }
