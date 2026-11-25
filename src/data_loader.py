"""Pemanggilan dan validasi dataset Seeds UCI."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "seeds_dataset.txt"
FALLBACK_PATH = PROJECT_ROOT / "seeds_dataset.txt"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/236/seeds.zip"

FEATURES = (
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "groove_length",
)
TARGET_COLUMN = "class_id"
VARIETY_COLUMN = "variety"
CLASS_MAP = {1: "Kama", 2: "Rosa", 3: "Canadian"}
LABEL_ORDER = ("Kama", "Rosa", "Canadian")

FEATURE_INFO = {
    "area": ("Luas", "Luas proyeksi kernel biji gandum."),
    "perimeter": ("Keliling", "Keliling batas kernel biji gandum."),
    "compactness": ("Kekompakan (compactness)", "Ukuran kekompakan kernel."),
    "kernel_length": ("Panjang kernel", "Panjang kernel biji gandum."),
    "kernel_width": ("Lebar kernel", "Lebar kernel biji gandum."),
    "asymmetry_coefficient": (
        "Koefisien asimetri",
        "Koefisien yang menggambarkan tingkat asimetri kernel.",
    ),
    "groove_length": (
        "Panjang alur kernel",
        "Panjang alur pada permukaan kernel.",
    ),
}


def _read_whitespace_data(source: str | Path | io.BytesIO) -> pd.DataFrame:
    frame = pd.read_csv(source, sep=r"\s+", header=None)
    if frame.shape[1] != len(FEATURES) + 1:
        raise ValueError(
            "Dataset Seeds harus memiliki 8 kolom (7 fitur dan 1 label); "
            f"ditemukan {frame.shape[1]} kolom."
        )
    frame.columns = [*FEATURES, TARGET_COLUMN]
    for column in [*FEATURES, TARGET_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        counts = frame.isna().sum()
        invalid = counts[counts > 0].to_dict()
        raise ValueError(f"Dataset mengandung nilai kosong atau nonnumerik: {invalid}")
    if not frame[TARGET_COLUMN].isin(CLASS_MAP).all():
        labels = sorted(frame[TARGET_COLUMN].unique().tolist())
        raise ValueError(f"Label kelas harus berupa 1, 2, atau 3; ditemukan {labels}.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    frame[VARIETY_COLUMN] = frame[TARGET_COLUMN].map(CLASS_MAP)
    return frame


def load_dataset(path: str | Path | None = None, allow_download: bool = True) -> pd.DataFrame:
    """Muat file lokal dahulu; arsip resmi UCI menjadi fallback terakhir."""
    if path is not None:
        selected_path = Path(path)
        if not selected_path.is_file():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {selected_path}")
        return _read_whitespace_data(selected_path)

    for candidate in (DATA_PATH, FALLBACK_PATH):
        if candidate.is_file():
            return _read_whitespace_data(candidate)

    if not allow_download:
        raise FileNotFoundError(
            "Dataset lokal tidak ditemukan. Sediakan data/seeds_dataset.txt "
            "atau izinkan pengunduhan arsip UCI sebagai fallback."
        )
    try:
        with urlopen(UCI_ARCHIVE_URL, timeout=30) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            with archive.open("seeds_dataset.txt") as dataset_file:
                return _read_whitespace_data(dataset_file)
    except Exception as exc:
        raise RuntimeError(
            "Dataset lokal tidak tersedia dan arsip UCI gagal diunduh. "
            "Letakkan seeds_dataset.txt di folder data/."
        ) from exc


def dataset_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Ringkasan pemeriksaan tanpa menghapus atau mengubah baris data."""
    features = frame.loc[:, FEATURES]
    return {
        "rows": len(frame),
        "feature_count": len(FEATURES),
        "column_count": len(FEATURES) + 1,
        "dtypes": frame.loc[:, [*FEATURES, TARGET_COLUMN]].dtypes.astype(str),
        "missing_values": frame.loc[:, [*FEATURES, TARGET_COLUMN]].isna().sum(),
        "duplicate_rows": int(frame.duplicated(subset=[*FEATURES, TARGET_COLUMN]).sum()),
        "class_distribution": frame[VARIETY_COLUMN].value_counts().reindex(LABEL_ORDER, fill_value=0),
        "feature_statistics": features.describe().T,
    }
