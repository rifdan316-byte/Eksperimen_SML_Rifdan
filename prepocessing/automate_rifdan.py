import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(raw_data_path, output_dir):
    print("Memulai proses preprocessing data...")
    
    # 1. Load Dataset
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"File data mentah tidak ditemukan di: {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    
    # 2. Handling Missing Values (Sesuaikan dengan notebook eksperimen Anda)
    # Contoh standar untuk dataset heart disease jika ada nulls
    df = df.dropna()
    
    # 3. Pisahkan Fitur (X) dan Target (y)
    # Asumsi kolom target bernama 'target' atau 'HeartDisease' (sesuaikan dengan dataset Anda)
    target_column = 'target' if 'target' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 4. Split Dataset (Train 80%, Test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Feature Scaling (Opsional, sesuaikan dengan kebutuhan eksperimen)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 6. Membuat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. Simpan hasil preprocessing ke file CSV
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Preprocessing selesai! File disimpan di direktori: {output_dir}")

if __name__ == "__main__":
    # Menentukan path secara relatif terhadap root repositori kriteria 1
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, 'data_raw', 'heart_disease.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'preprocessing', 'data_processed')
    
    preprocess_data(RAW_PATH, OUTPUT_PATH)