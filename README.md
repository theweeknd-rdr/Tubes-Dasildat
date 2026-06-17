# Prediksi Depresi Mahasiswa

Aplikasi Streamlit untuk memprediksi label `depresion` pada data mahasiswa menggunakan model machine learning yang tersimpan di folder `model/`.

## Model yang Digunakan

File model utama:

```text
model/knn_depression_model.joblib
```

Model tersebut adalah model KNN yang membutuhkan fitur input dalam bentuk numerik atau hasil encoding.

## Fitur Input

Aplikasi menerima fitur berikut dari user:

- `Gender`
- `Age`
- `Profession`
- `Academic Pressure`
- `CGPA`
- `Study Satisfaction`
- `Job Satisfaction`
- `Sleep Duration`
- `Dietary Habits`
- `Degree`
- `Have you ever had suicidal thoughts ?`
- `Work/Study Hours`
- `Financial Stress`
- `Family History of Mental Illness`
- `City_freq_City`

Model juga membutuhkan fitur `Work Pressure`. Karena konteks aplikasi adalah mahasiswa, aplikasi mengisi `Work Pressure` secara otomatis dengan nilai `0`.

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

```powershell
cd C:\tubes_klasifikas_kelompok7
venv\Scripts\Activate.ps1
streamlit run appBC_Batch.py
```

Jika dependency belum terpasang:

```powershell
pip install -r requirements.txt
```

## Deploy ke Streamlit Community Cloud

Gunakan konfigurasi berikut:

```text
Repository: theweeknd-rdr/Tubes-Dasildat
Branch: Portofolio
Main file path: appBC_Batch.py
```

## Catatan Penting

Aplikasi ini dibuat untuk kebutuhan pembelajaran dan demonstrasi model machine learning. Hasil prediksi bukan diagnosis medis.
