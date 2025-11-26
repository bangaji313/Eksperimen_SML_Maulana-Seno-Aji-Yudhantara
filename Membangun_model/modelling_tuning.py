import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 1. KONFIGURASI DAGSHUB
DAGSHUB_USERNAME = "bangaji313" 
REPO_NAME = "Eksperimen_SML_Maulana-Seno-Aji-Yudhantara"

print("Menghubungkan ke DagsHub...")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("Eksperimen Churn Prediction - Advanced")

# 2. LOAD DATA
# Ambil data yang sudah dibersihkan dari Kriteria 1
data_path = '../preprocessing/customer_churn_cleaned.csv'

if not os.path.exists(data_path):
    print(f"Error: File {data_path} tidak ditemukan. Jalankan automate script dulu!")
    exit()

print("Loading dataset...")
df = pd.read_csv(data_path)

# Pisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. HYPERPARAMETER TUNING (Syarat Skilled)
print("Memulai Hyperparameter Tuning...")

# Kita pakai Random Forest
rf = RandomForestClassifier(random_state=42)

# Grid Search (Mencari kombinasi terbaik)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Parameter Terbaik: {best_params}")

# 4. EVALUASI MODEL
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Akurasi Model: {accuracy}")

# 5. LOGGING KE MLFLOW/DAGSHUB (Syarat Advanced)
print("Mengirim report ke DagsHub...")

with mlflow.start_run():
    # A. Log Parameters (Bumbu racikan)
    mlflow.log_params(best_params)
    
    # B. Log Metrics (Nilai rapor)
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    # C. Log Model (Simpan masakan jadi)
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # D. Log Artefak Tambahan 1: Confusion Matrix Image (Syarat Advanced)
    print("Membuat Artefak 1: Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # E. Log Artefak Tambahan 2: Feature Importance (Syarat Advanced)
    print("Membuat Artefak 2: Feature Importance...")
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    print("Selesai! Cek DagsHub.")