import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Full paths
base_dir = r"C:\Users\Sulaiman Dalhatu\AI_Student_Project"
dropout_path = os.path.join(base_dir, "data", "processed", "dropout_ready.csv")

# Load CSV
if not os.path.exists(dropout_path):
    raise FileNotFoundError(f"Processed dataset not found at {dropout_path}. Make sure Phase 5 fix script ran correctly.")

df = pd.read_csv(dropout_path, sep=';', engine='python', quotechar='"')
print("Columns loaded:", df.columns.tolist())
print("Number of rows before fixing Target:", len(df))

# Ensure Target exists
if 'Target' not in df.columns:
    for col in df.columns:
        if 'Target' in col:
            df.rename(columns={col: 'Target'}, inplace=True)
            print(f"Renamed '{col}' to 'Target'")
            break
    else:
        df['Target'] = 0
        print("Target column not found. Created a new Target column with default 0 values.")

# Fill missing Target values
df['Target'] = df['Target'].fillna(0)

# Drop rows with essential NaNs (optional: add more essential columns if needed)
essential_cols = ['Target']  # you can add more columns if your model requires
df = df.dropna(subset=essential_cols)

print("Number of rows after fixing Target and dropping essential NaNs:", len(df))

if len(df) == 0:
    raise ValueError("No data available for training after fixing Target values. Check your CSV.")

# --- Features & Labels ---
X = df.drop(columns=['Target'])
y = df['Target']

# Optional: convert categorical columns to numeric if needed
X = pd.get_dummies(X, drop_first=True)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train/Test split successful.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Your explainability code goes here ---
# e.g., SHAP, LIME, or feature importance
