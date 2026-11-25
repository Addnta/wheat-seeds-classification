# Klasifikasi Varietas Biji Gandum

Aplikasi akademik Streamlit untuk mengeksplorasi Seeds Dataset UCI dan membandingkan K-Nearest Neighbor, Decision Tree, serta Gaussian Naive Bayes.

## Dataset

Aplikasi menggunakan file `data/seeds_dataset.txt`, salinan dataset Seeds dari [UCI Machine Learning Repository (ID 236)](https://archive.ics.uci.edu/dataset/236/seeds). Dataset memuat tujuh fitur geometris dan label kelas 1 (Kama), 2 (Rosa), atau 3 (Canadian). Loader membaca file lokal terlebih dahulu; arsip resmi UCI hanya diunduh sebagai fallback jika file lokal tidak tersedia.

## Metodologi

- KNN: `n_neighbors=5`, bobot uniform, jarak Minkowski dengan `p=2`; `StandardScaler` ada di dalam `Pipeline`.
- Decision Tree: kriteria Gini, tanpa batas kedalaman, `random_state=42`.
- Gaussian Naive Bayes: `var_smoothing=1e-9`.
- Evaluasi: Stratified 5-Fold Cross-Validation, `shuffle=True`, `random_state=42`.
- Tabel mean dan simpangan baku berasal dari metrik masing-masing fold. Confusion matrix dan classification report berasal dari prediksi out-of-fold yang ditampilkan terpisah.
- Model prediksi akhir baru dilatih pada seluruh data setelah evaluasi CV dihitung.

## Struktur proyek

```text
app.py
pages/
  1_Dashboard.py
  2_Dataset.py
  3_Visualisasi.py
  4_Evaluasi_Model.py
  5_Prediksi.py
  6_Tentang.py
data/seeds_dataset.txt
src/
  app_data.py
  data_loader.py
  evaluation.py
  models.py
  prediction.py
  preprocessing.py
  ui.py
tests/test_foundation.py
requirements.txt
```

## Instalasi dan menjalankan lokal

Gunakan Python 3.10 atau lebih baru.

```bash
python -m venv .venv
```

Aktifkan lingkungan virtual, lalu jalankan:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit. Dataset sudah disertakan sehingga aplikasi tidak memerlukan koneksi internet saat berjalan normal.

## Pengujian

Jalankan tes fondasi dengan pustaka standar Python:

```bash
python -m unittest discover -s tests -v
```

Tes mencakup pemuatan dan validasi data, pemisahan fitur-target, konfigurasi model, evaluasi CV dan OOF, serta prediksi dan validasi rentang input.

## Mengunggah ke GitHub

1. Buat repositori kosong di GitHub.
2. Dari folder proyek, inisialisasi Git bila belum ada, lalu tambahkan berkas proyek dan buat commit.
3. Tambahkan alamat repositori GitHub sebagai remote dan dorong branch utama.
4. Pastikan `data/seeds_dataset.txt`, `requirements.txt`, dan `app.py` ikut terunggah.

## Deploy ke Streamlit Community Cloud

1. Dorong proyek ke repositori GitHub yang dapat diakses akun Streamlit.
2. Di Streamlit Community Cloud, pilih **Create app**, repositori, branch, dan `app.py` sebagai main file.
3. Deploy aplikasi. Dependensi akan dipasang dari `requirements.txt`.

Deployment belum dilakukan dari lingkungan ini.
