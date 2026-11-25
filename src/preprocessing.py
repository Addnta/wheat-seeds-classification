"""Pemisahan fitur dan target untuk seluruh alur pemodelan."""

from __future__ import annotations

import pandas as pd

from src.data_loader import FEATURES, VARIETY_COLUMN


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pastikan hanya tujuh fitur masuk ke X dan varietas menjadi target y."""
    missing = [column for column in [*FEATURES, VARIETY_COLUMN] if column not in frame]
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {missing}")
    X = frame.loc[:, FEATURES].copy()
    y = frame[VARIETY_COLUMN].copy()
    if X.shape[1] != 7:
        raise ValueError(f"Diperlukan 7 fitur, ditemukan {X.shape[1]}.")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Fitur dan target tidak boleh mengandung nilai kosong.")
    return X, y
