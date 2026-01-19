
# scripts/phase3_data_cleaning_preprocessing.py
import pandas as pd
import numpy as np
import os

raw_path = '../data/raw/student_dropout_performance.csv'
cleaned_path = '../data/cleaned/student_cleaned.csv'

os.makedirs('../data/cleaned/', exist_ok=True)

df = pd.read_csv(raw_path, sep=';')

# Example cleaning: handle missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

# Remove extreme outliers (example)
numerical_cols = df.select_dtypes(include='number').columns
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numerical_cols] < (Q1 - 1.5*IQR)) | (df[numerical_cols] > (Q3 + 1.5*IQR))).any(axis=1)]

# Save cleaned data
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved to: {cleaned_path}")
