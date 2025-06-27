import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/umkm_dataset.csv')

# Contoh dataset (Anda perlu menyiapkan dataset asli)
# Format dataset harus mengandung fitur-fitur keuangan dan label kesehatan
# Misal: pendapatan, pengeluaran, hutang, aset, dll dan label 'kesehatan'

# Pra-pemrosesan data
X = data.drop('kesehatan', axis=1)
y = data['kesehatan']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save model and scaler
joblib.dump(model, 'models/umkm_model.pkl')
joblib.dump(scaler, 'models/umkm_scaler.pkl')
print("Model dan scaler telah disimpan!")