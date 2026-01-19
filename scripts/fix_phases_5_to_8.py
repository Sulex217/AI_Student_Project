import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ================================
# PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cleaned_path = os.path.join(BASE_DIR, "data", "cleaned", "student_cleaned.csv")
processed_dir = os.path.join(BASE_DIR, "data", "processed")
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

dropout_ready_path = os.path.join(processed_dir, "dropout_ready.csv")
model_path = os.path.join(models_dir, "dropout_model.pkl")

# ================================
# PHASE 5 — LOAD & FIX DATA
# ================================
print("🔹 Loading cleaned dataset...")
df = pd.read_csv(cleaned_path, sep=';', engine='python')

print("Columns loaded:", df.columns.tolist())
print("Initial number of rows:", len(df))

# --- Fix broken Target column names ---
for col in df.columns:
    if "Target" in col:
        df.rename(columns={col: "Target"}, inplace=True)

# Ensure Target exists
if "Target" not in df.columns:
    raise KeyError("❌ Target column not found. Please check your CSV header.")

# --- Fix Target values ---
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

df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
df["Target"] = df["Target"].fillna(0).astype(int)

# --- Drop rows with all NaNs ---
df.dropna(how="all", inplace=True)

print("Rows after fixing Target:", len(df))

# --- Save processed dataset ---
df.to_csv(dropout_ready_path, index=False, sep=';')
print(f"✅ Dropout dataset saved to: {dropout_ready_path}")

# ================================
# PHASE 6 — MODEL DEVELOPMENT
# ================================
print("\n🔹 Starting model development...")

df = pd.read_csv(dropout_ready_path, sep=';', engine='python')

print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# Separate features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# --- Encode categorical variables ---
X = pd.get_dummies(X, drop_first=True)

# --- Ensure no NaNs ---
X.fillna(0, inplace=True)

if X.shape[0] == 0:
    raise ValueError("❌ No data available for training after preprocessing.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# ================================
# PHASE 7 — MODEL EVALUATION
# ================================
print("\n🔹 Evaluating model...")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# PHASE 8 — EXPLAINABILITY PREP
# ================================
print("\n🔹 Preparing explainability data...")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

explain_path = os.path.join(processed_dir, "feature_importance.csv")
feature_importance.to_csv(explain_path, index=False)

print(f"✅ Feature importance saved to: {explain_path}")

print("\n🎉 ALL PHASES 5–8 FIXED AND COMPLETED SUCCESSFULLY!")
