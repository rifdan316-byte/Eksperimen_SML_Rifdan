import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import imblearn.under_sampling
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    """Membaca data mentah dari path menggunakan absolute path atau relative path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")
    return pd.read_csv(file_path)

def preprocess_and_split(df):
    """
    Menjalankan pembersihan data, encoding, pemisahan fitur, 
    dan melakukan split & scaling secara aman tanpa data leakage.
    """
    # 1. Mengatasi data duplikat
    df = df.drop_duplicates()
    
    # 2. Lakukan One-Hot Encoding pada fitur kategorikal teks agar bisa di-scale
    df_encoded = pd.get_dummies(df, columns=['Student_Type', 'Month'], drop_first=True)
    
    # 3. Pisahkan Fitur (X) dan Target (y) berdasarkan nama kolom yang tepat
    X = df_encoded.drop(columns=['Stress_Level']) 
    y = df_encoded['Stress_Level']
    
    # 4. Split Dataset terlebih dahulu (Proporsi 80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Penskalaan (Scaling) Fitur secara terisolasi untuk mencegah leakage
    scaler = StandardScaler()
    
    # Fit & Transform hanya pada data training
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # Transform saja pada data testing menggunakan parameter dari data training
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

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
    # Menggunakan Absolute Path agar aman dijalankan dari direktori terminal mana pun
    RAW_DATA_PATH = r"C:\Users\LENOVO\Documents\MSML\Eksperimen_SML_Rifdan\data_raw\student-lifestyle-and-stress-dataset.csv"
    OUTPUT_DIR = r"C:\Users\LENOVO\Documents\MSML\Eksperimen_SML_Rifdan\preprocessing\student_preprocessing"
    
    print("=== Memulai Pipeline Otomatisasi Preprocessing ===")
    
    # 1. Load Data
    raw_df = load_data(RAW_DATA_PATH)
    
    # 2. Preprocess, One-Hot Encode, Split, dan Scale (Aman dari Leakage & ValueError)
    X_train, X_test, y_train, y_test = preprocess_and_split(raw_df)
    
    # 3. Handle Imbalanced Data HANYA pada Data Train
    X_train_res, y_train_res = handle_imbalanced_data(X_train, y_train)
    
    # 4. Simpan Hasil Akhir yang Siap Dilatih ke Folder Target
    save_preprocessed_data(X_train_res, X_test, y_train_res, y_test, OUTPUT_DIR)
    
    print("=== Pipeline Selesai Terbaca Tanpa Error ===")