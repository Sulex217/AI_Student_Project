import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ================================
# PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_path = os.path.join(BASE_DIR, "data", "processed", "dropout_ready.csv")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "dropout_model.pkl")

# ================================
# LOAD DATA
# ================================
print("🔹 Loading processed dataset...")
df = pd.read_csv(processed_path, sep=';', engine='python')

print("Columns loaded:", df.columns.tolist())
print("Number of rows:", len(df))

# ================================
# FIX TARGET COLUMN
# ================================
# Normalize Target column name
for col in df.columns:
    if "Target" in col:
        df.rename(columns={col: "Target"}, inplace=True)

if "Target" not in df.columns:
    raise KeyError("❌ Target column not found in dataset.")

# Clean Target values
df["Target"] = df["Target"].astype(str).str.strip().str.lower()
df["Target"] = df["Target"].replace({
    "dropout": 1,
    "enrolled": 0,
    "graduate": 0,
    "0": 0,
    "1": 1,
    "": np.nan,
    "nan": np.nan
})

df["Target"] = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(int)

# ================================
# PREPARE FEATURES
# ================================
X = df.drop("Target", axis=1)
y = df["Target"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Fill remaining NaNs
X.fillna(0, inplace=True)

if X.shape[0] == 0:
    raise ValueError("❌ No data available for training after preprocessing.")

# ================================
# TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# TRAIN MODEL
# ================================
print("🔹 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================================
# EVALUATE MODEL
# ================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# SAVE MODEL
# ================================
joblib.dump(model, model_path)
print(f"💾 Model saved to: {model_path}")
