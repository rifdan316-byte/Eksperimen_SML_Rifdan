"""
Skrip Otomasi Pipeline Machine Learning (End-to-End)
Proyek: Student Lifestyle and Stress Prediction
Nama Mahasiswa: Rifdan
Kriteria Dicoding: Proyek Akhir (Kriteria 3 - Automate Script)
"""

import os
import pickle
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_automation_pipeline():
    print("="*60)
    print("STARTING MACHINE LEARNING AUTOMATION PIPELINE - RIFDAN")
    print("="*60)

    # 1. PENYELARASAN DIREKTORI & MEMUAT DATASET
    # Memastikan skrip berjalan dari folder proyek utama
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data_raw", "student-lifestyle-and-stress-dataset.csv")
    
    if not os.path.exists(data_path):
        # Fallback jika dijalankan langsung di root tanpa subfolder data_raw
        data_path = os.path.join(base_dir, "student-lifestyle-and-stress-dataset.csv")
        
    print(f"[1/6] Memuat dataset dari: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"      -> Sukses! Ukuran dataset awal: {df.shape[0]} baris, {df.shape[1]} kolom.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset. Pesan kesalahan: {e}")
        return

    # 2. PEMISAHAN FITUR DAN TARGET
    target_column = 'Stress_Level'
    if target_column not in df.columns:
        print(f"[ERROR] Kolom target '{target_column}' tidak ditemukan dalam dataset!")
        return
        
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. PRAPEMROSESAN DATA & PIPELINE REKAYASA FITUR (PREPROCESSING)
    print("[2/6] Membangun pipeline prapemrosesan data otomatis...")
    
    # Identifikasi kolom numerik dan kategorikal secara dinamis
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"      -> Fitur Numerik   : {numeric_features}")
    print(f"      -> Fitur Kategorikal: {categorical_features}")

    # Pipeline untuk fitur numerik: Imputasi Mean + Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal: Imputasi Mode + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Menggabungkan preprocessor menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. DATA SPLITTING & HANDLING CLASS IMBALANCE
    print("[3/6] Melakukan pembagian data uji (Train-Test Split 80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"      -> Sebelum Resampling - Distribusi Kelas Train: {dict(pd.Series(y_train).value_counts())}")
    
    print("[4/6] Menangani masalah ketidakseimbangan kelas (Downsampling)...")
    # Karena data awal di-preprocess lewat pipeline, kita transformasikan X_train sementara untuk resampling
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Menerapkan Random Under Sampler
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_transformed, y_train)
    
    print(f"      -> Setelah Resampling - Distribusi Kelas Train: {dict(pd.Series(y_train_resampled).value_counts())}")

    # 5. PELATIHAN MODEL (RANDOM FOREST CLASSIFIER)
    print("[5/6] Melatih model Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    print("      -> Model berhasil dilatih.")

    # 6. EVALUASI MODEL SECARA RIGORIS
    print("[6/6] Mengevaluasi model pada data uji...")
    y_pred = model.predict(X_test_transformed)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("\n" + "="*45)
    print("         RINGKASAN EVALUASI MODEL          ")
    print("="*45)
    print(f" Akurasi  : {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall   : {recall:.4f}")
    print(f" F1-Score : {f1:.4f}")
    print("="*45)
    print("\nLaporan Klasifikasi Detail:")
    print(classification_report(y_test, y_pred))

    # 7. EKSPOR ARTIFAK MODEL AKHIR
    # Menyimpan model dan objek preprocessor untuk deployment masa depan
    model_export_path = os.path.join(base_dir, "saved_models")
    os.makedirs(model_export_path, exist_ok=True)
    
    # Mengekspor pipeline preprocessor dan model utama menggunakan Pickle (.pkl)
    pipeline_file = os.path.join(model_export_path, "final_pipeline_model.pkl")
    with open(pipeline_file, 'wb') as f:
        pickle.dump({'preprocessor': preprocessor, 'model': model}, f)
        
    print(f"\n[SUKSES] Seluruh pipeline otomatis selesai dijalankan!")
    print(f"[SUKSES] Model akhir dan konfigurasi pipeline berhasil diekspor ke: {pipeline_file}")
    print("="*60)

if __name__ == "__main__":
    run_automation_pipeline()