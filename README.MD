# Panduan Instalasi dan Menjalankan Aplikasi FinDetect

## Persyaratan Sistem
- Python 3.9 atau lebih baru
- pip (package installer for Python)
- Virtual environment (direkomendasikan)

## Langkah-langkah Instalasi


### 1. Setup Virtual Environment (Rekomendasi)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**MacOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan Dataset dan Model
- Pastikan file dataset berada di `data/umkm_dataset.csv`
- Jalankan script training untuk membuat model:
```bash
python model_training.py
```

## Menjalankan Aplikasi

### Mode Pengembangan
```bash
python app.py
```
Aplikasi akan berjalan di `http://localhost:5000`

### Mode Produksi (Menggunakan Waitress)
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

## Struktur Direktori
```
finance-detector-umkm/
├── app.py                # Aplikasi utama Flask
├── model_training.py     # Script pelatihan model
├── requirements.txt      # Dependensi Python
├── data/
│   └── umkm_dataset.csv  # Dataset contoh
├── models/               # Model ML yang sudah trained
├── static/               # File static (CSS, gambar)
└── templates/            # Template HTML
```

## Akses Halaman
- Landing Page: `http://localhost:5000`
- Halaman Prediksi: `http://localhost:5000/predict`
- Tentang Kami: `http://localhost:5000/about`
- Fitur: `http://localhost:5000/features`

## Troubleshooting

### Jika ada masalah instalasi:
1. Pastikan Python versi terbaru terinstall
2. Gunakan virtual environment
3. Untuk error spesifik, coba install package satu per satu:
```bash
pip install flask pandas scikit-learn numpy joblib
```

### Jika model tidak berjalan:
1. Pastikan `model_training.py` sudah dijalankan
2. Cek file `models/umkm_model.pkl` dan `models/umkm_scaler.pkl` ada
