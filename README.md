# Breast Cancer Prediction App

Aplikasi Streamlit untuk memprediksi breast cancer menggunakan model machine learning yang disimpan dalam folder `model/`.

## Fitur

- Prediksi manual berdasarkan input fitur.
- Prediksi batch menggunakan file CSV atau Excel.
- Pemilihan model dari file `.joblib` atau `.pkl` di folder `model/`.
- Download hasil prediksi batch dalam format CSV.

## Struktur Proyek

```text
tubes_klasifikas_kelompok7/
├── appBC_Batch.py
├── appManual.py
├── requirements.txt
├── model/
│   ├── knn_depression_model.joblib
│   ├── modelJb_SVM.joblib
│   └── modelJb_SVM-HPO.joblib
└── README.md
```

## Cara Menjalankan di Lokal

Aktifkan virtual environment, lalu jalankan aplikasi:

```powershell
venv\Scripts\Activate.ps1
streamlit run appBC_Batch.py
```

Jika dependency belum terpasang:

```powershell
pip install -r requirements.txt
```

## Deploy ke Streamlit Community Cloud

1. Push project ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub.
4. Isi main file path dengan:

```text
appBC_Batch.py
```

5. Deploy aplikasi.

## Catatan Dataset

Model menggunakan fitur berikut:

- `concave points_worst`
- `perimeter_worst`
- `concave points_mean`
- `radius_worst`
- `perimeter_mean`

Untuk prediksi CSV atau Excel, pastikan nama kolom pada file upload sama persis dengan daftar fitur tersebut.
