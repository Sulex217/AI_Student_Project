import os
import pandas as pd

# Paths
cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "student_cleaned.csv")
processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

dropout_output_path = os.path.join(processed_dir, "dropout_ready.csv")
performance_output_path = os.path.join(processed_dir, "performance_ready.csv")

# Load cleaned dataset
df = pd.read_csv(cleaned_path, sep=';', engine='python', quotechar='"')
print("Columns loaded:", df.columns.tolist())
print("Number of rows:", len(df))

# === Dropout dataset ===
# Keep Target column and drop rows with missing Target
if 'Target' not in df.columns:
    raise KeyError("Target column not found in cleaned dataset.")

df_dropout = df.dropna(subset=['Target']).copy()
print("Rows after dropping NaN Target:", len(df_dropout))

# Save Dropout CSV
df_dropout.to_csv(dropout_output_path, index=False, sep=';')
print("Dropout dataset saved to:", dropout_output_path)

# === Performance dataset ===
# Example: you can do additional performance-related feature engineering
df_performance = df.copy()  # modify if needed
df_performance.to_csv(performance_output_path, index=False, sep=';')
print("Performance dataset saved to:", performance_output_path)
