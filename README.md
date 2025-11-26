# Eksperimen Sistem Machine Learning - Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12.7-blue?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![DagsHub](https://img.shields.io/badge/DagsHub-Tracking-black?style=for-the-badge&logo=dagshub&logoColor=white)

Repository ini merupakan bagian dari **Kriteria 1 (Eksperimen Data)** dan **Kriteria 2 (Membangun Model)** untuk Submission Proyek Akhir Kelas **"Membangun Sistem Machine Learning"** di Dicoding Indonesia.

Proyek ini berfokus pada pengembangan pipeline *Machine Learning* untuk memprediksi **Customer Churn** (pelanggan yang berhenti berlangganan), mulai dari eksperimen data manual, otomatisasi preprocessing, hingga pelacakan eksperimen model secara online.

## 📋 Daftar Isi
- [Tentang Proyek](#-tentang-proyek)
- [Dataset](#-dataset)
- [Struktur Repository](#-struktur-repository)
- [Fitur Utama](#-fitur-utama)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil Eksperimen](#-hasil-eksperimen)

## 📖 Tentang Proyek
Tujuan utama dari repository ini adalah mendemonstrasikan penerapan prinsip **MLOps** dalam tahap awal pengembangan model, yaitu:
1.  **Reproducibility:** Memastikan proses pengolahan data dapat diulang dengan hasil yang sama.
2.  **Automation:** Menggunakan script Python dan GitHub Actions untuk mengotomatisasi *data cleaning*.
3.  **Experiment Tracking:** Melacak metrik performa model dan parameter menggunakan MLflow yang terintegrasi dengan DagsHub.

## 💾 Dataset
Dataset yang digunakan adalah **Customer Churn Dataset** yang bersumber dari Kaggle.
- **Sumber:** [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
- **Tipe Data:** Tabular (CSV)
- **Target:** Klasifikasi Biner (Churn: 0 atau 1)

## 📂 Struktur Repository
```text
Eksperimen_SML_Maulana-Seno-Aji-Yudhantara/
├── .github/workflows/
│   └── data_pipeline.yml        # Workflow GitHub Actions untuk otomatisasi
├── data_raw/                    # Folder penyimpanan dataset mentah
├── preprocessing/               # Folder penyimpanan dataset hasil cleaning
├── Membangun_model/             # Folder training & eksperimen model
│   ├── mlruns/                  # Log lokal MLflow
│   └── modelling_tuning.py      # Script training dengan Hyperparameter Tuning
├── automate_Maulana-Seno-Aji-Yudhantara.py  # Script Python untuk preprocessing otomatis
├── Eksperimen_Maulana-Seno-Aji-Yudhantara.ipynb # Notebook eksperimen (EDA & Manual)
├── requirements.txt             # Daftar library yang dibutuhkan
└── README.md                    # Dokumentasi proyek
```

## 🚀 Fitur Utama

### 1. Otomatisasi Preprocessing (Level Advanced)

Repository ini dilengkapi dengan workflow **GitHub Actions** (`data_pipeline.yml`).  
Setiap kali ada *push* ke branch `main`, sistem akan otomatis:

- Menjalankan script `automate_Maulana-Seno-Aji-Yudhantara.py`
- Membersihkan data raw (*dropna*, penghapusan kolom ID, encoding)
- Menyimpan hasil bersih ke folder `preprocessing/`
- Melakukan auto-commit hasil preprocessing ke repository

---

### 2. Hyperparameter Tuning & Logging (Level Advanced)

Script `modelling_tuning.py` digunakan untuk:

- Melakukan **Grid Search** pada algoritma *Random Forest*
- Mencatat parameter terbaik (`max_depth`, `n_estimators`, dll) ke **DagsHub**
- Menyimpan metrik evaluasi:  
  **Accuracy**, **Precision**, **Recall**, **F1-Score**
- Menyimpan artefak visual:
  - Confusion Matrix  
  - Feature Importance

---

## 💻 Cara Menjalankan

### Prasyarat

- Python **3.12.7**
- Akun **DagsHub** (untuk tracking eksperimen)

---

## Instalasi

### Clone Repository
```bash
git clone https://github.com/bangaji313/Eksperimen_SML_Maulana-Seno-Aji-Yudhantara.git
cd Eksperimen_SML_Maulana-Seno-Aji-Yudhantara
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Menjalankan Preprocessing (Lokal)
```bash
python automate_Maulana-Seno-Aji-Yudhantara.py
```
Output: File hasil preprocessing berada di folder preprocessing/.
### Menjalankan Training & Tracking
- Setup Environment Variable DagsHub
  ```bash
  export MLFLOW_TRACKING_URI="https://dagshub.com/bangaji313/Eksperimen_SML_Maulana-Seno-Aji-Yudhantara.mlflow"
  export MLFLOW_TRACKING_USERNAME="bangaji313"
  export MLFOW_TRACKING_PASSWORD="<Token_DagsHub_Anda>"
  ```
- Jalankan Script Training
  ```bash
  cd Membangun_model
  python modelling_tuning.py
  ```
---
## 📊 Hasil Eksperimen
Hasil eksperimen dapat dilihat secara real-time melalui dashboard DagsHub proyek ini.
- Metode: Random Forest Classifier dengan Grid Search CV
- Metrik: Akurasi tinggi pada data testing
- Artifacts: Confusion Matrix & Feature Importance untuk analisis prediksi
---
## 👤 Author
> Maulana Seno Aji Yudhantara
> Mahasiswa Informatika – Institut Teknologi Nasional Bandung


