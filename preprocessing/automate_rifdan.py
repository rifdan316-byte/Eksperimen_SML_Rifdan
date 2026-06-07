import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def run_preprocessing():
    print("=== Memulai Automasi Preprocessing Data (Rifdan) ===")
    
    # 1. Load Data
    raw_path = '../data_raw/student-lifestyle-and-stress-dataset.csv'
    if not os.path.exists(raw_path):
        print(f"Error: File tidak ditemukan di {raw_path}")
        return
        
    df = pd.read_csv(raw_path)
    
    # 2. Clean Target
    df = df.dropna(subset=['Stress_Level'])
    df['Stress_Level'] = df['Stress_Level'].astype(int)
    
    # 3. Handle Anomalies
    if 'Attendance' in df.columns:
        df.loc[(df['Attendance'] < 0) | (df['Attendance'] > 100), 'Attendance'] = np.nan
        
    # 4. Imputation
    num_cols = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Attendance', 'Exam_Pressure', 'Family_Support', 'Month']
    cat_cols = ['Student_Type']
    
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    # 5. IQR Capping Outliers
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
    # 6. Encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # 7. Split Data
    X = df.drop(columns=['Stress_Level'])
    y = df['Stress_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 8. Feature Scaling
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # 9. Save Output
    output_folder = 'student-lifestyle-preprocessing'
    os.makedirs(output_folder, exist_ok=True)
    
    X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
    X_test.to_csv(f'{output_folder}/X_test.csv', index=False)
    y_train.to_csv(f'{output_folder}/y_train.csv', index=False)
    y_test.to_csv(f'{output_folder}/y_test.csv', index=False)
    
    print("✔ Automasi Selesai! Dataset hasil preprocessing berhasil diperbarui.")

if __name__ == "__main__":
    run_preprocessing()