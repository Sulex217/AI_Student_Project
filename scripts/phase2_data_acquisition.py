# scripts/phase2_data_acquisition.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Relative path from scripts/ folder
data_path = '../data/raw/student_dropout_performance.csv'
df = pd.read_csv(data_path, sep=';')  # CSV uses semicolons

# Inspect data
print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())
