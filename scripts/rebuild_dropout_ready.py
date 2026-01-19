import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "raw", "student_dropout_performance.csv")
processed_path = os.path.join(BASE_DIR, "data", "processed", "dropout_ready.csv")

print("🔹 Loading raw dataset with tolerant parser...")

df = pd.read_csv(
    raw_path,
    sep=';',
    engine='python',
    on_bad_lines='skip'
)

print(f"✔ Loaded {len(df)} rows.")
print("📋 Columns found:")
print(df.columns.tolist())

# 🔍 Auto-detect target column
possible_targets = ['Target', 'target', 'Status', 'status', 'Outcome', 'outcome', 'Class', 'class']
target_col = None

for col in df.columns:
    if col.strip().lower() in [p.lower() for p in possible_targets]:
        target_col = col
        break

# If not found, assume last column
if target_col is None:
    target_col = df.columns[-1]
    print(f"⚠️ Target column not explicitly named — assuming last column: {target_col}")

# Rename to Target
df = df.rename(columns={target_col: 'Target'})

# Clean Target column
df['Target'] = df['Target'].astype(str).str.strip()

target_map = {
    'Graduate': 0,
    'Dropout': 1,
    'Enrolled': 0,
    '0': 0,
    '1': 1
}

df['Target'] = df['Target'].replace(target_map)
df['Target'] = pd.to_numeric(df['Target'], errors='coerce')

before = len(df)
df = df.dropna(subset=['Target'])
after = len(df)

print(f"✔ Rows after cleaning Target: {after} (removed {before - after})")

# Save cleaned dataset
df.to_csv(processed_path, index=False, sep=';')
print(f"✅ Clean dataset saved to: {processed_path}")
