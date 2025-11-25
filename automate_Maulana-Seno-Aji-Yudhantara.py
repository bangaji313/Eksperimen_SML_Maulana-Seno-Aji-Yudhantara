import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # 1. Definisikan path file
    input_path = 'data_raw/customer_churn_dataset-training-master.csv'
    output_path = 'preprocessing/customer_churn_cleaned.csv'
    
    # Cek apakah file input ada
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # 2. Cleaning Data
    # Hapus baris yang missing value 
    initial_rows = len(df)
    df = df.dropna()
    print(f"Menghapus {initial_rows - len(df)} baris data kosong.")
    
    # Hapus kolom CustomerID
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Kolom CustomerID dihapus.")
        
    # 3. Encoding (Mengubah Teks ke Angka)
    cat_columns = ['Gender', 'Subscription Type', 'Contract Length']
    
    le = LabelEncoder()
    for col in cat_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            print(f"Kolom {col} berhasil di-encode.")
            
    # 4. Simpan Hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sukses! Data bersih tersimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data()