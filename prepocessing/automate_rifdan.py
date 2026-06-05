import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    """Membaca data mentah dari path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Menjalankan tahapan pembersihan data dan pemisahan fitur secara otomatis."""
    # 1. Mengatasi data duplikat jika ada
    df = df.drop_duplicates()
    
    # 2. Pisahkan Fitur (X) dan Target (y) sesuai nama kolom di dataset Anda
    X = df.drop(columns=['Stress Level']) 
    y = df['Stress Level']
    
    # 3. Lakukan Scaling pada Fitur Numerik
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

def handle_imbalanced_data(X_train, y_train):
    """Melakukan Random Undersampling HANYA pada data training."""
    print("Melakukan Random Undersampling pada data training...")
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """Menyimpan hasil data split dan resampled ke folder tujuan."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, "X_train_ready.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test_ready.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train_ready.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test_ready.csv"), index=False)
    print(f"Sukses! Seluruh data hasil preprocessing disimpan di: {output_dir}")

if __name__ == "__main__":
    # Tentukan jalur file sesuai dengan struktur folder Eksperimen_SML_Rifdan Anda
    RAW_DATA_PATH = "data_raw/student_stress_dataset.csv"  # Sesuaikan dengan nama file asli Anda
    OUTPUT_DIR = "preprocessing/namadataset_preprocessing"
    
    print("=== Memulai Pipeline Otomatisasi Preprocessing ===")
    
    # 1. Load Data
    raw_df = load_data(RAW_DATA_PATH)
    
    # 2. Preprocess & Scale Fitur
    X, y = preprocess_data(raw_df)
    
    # 3. Split Dataset Terlebih Dahulu (Menghindari Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle Imbalanced Data HANYA pada Data Train
    X_train_res, y_train_res = handle_imbalanced_data(X_train, y_train)
    
    # 5. Simpan Hasil Akhir yang Siap Dilatih ke Folder Target
    save_preprocessed_data(X_train_res, X_test, y_train_res, y_test, OUTPUT_DIR)
    
    print("=== Pipeline Selesai Terbaca Tanpa Error ===")