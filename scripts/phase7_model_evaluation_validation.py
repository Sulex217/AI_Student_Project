import pandas as pd

# Path to the dropout dataset
processed_path = r"C:\Users\Sulaiman Dalhatu\AI_Student_Project\data\processed\dropout_ready.csv"


# Load the processed dataset
df = pd.read_csv(processed_path, sep=';', engine='python', quotechar='"')

print("Columns loaded:", list(df.columns))
print("Number of rows before filling Target:", len(df))

# Fill missing Target values with 0 (or 1 depending on your preference)
if 'Target' in df.columns:
    df['Target'] = df['Target'].fillna(0)
else:
    raise KeyError("Target column not found in processed dataset.")

print("Number of rows after filling missing Target values:", len(df))

# Now you can proceed with model evaluation
# Example: splitting features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Simple check
if y.isna().sum() > 0:
    raise ValueError("There are still missing values in the Target column.")

print("Target column cleaned and ready for model evaluation.")
