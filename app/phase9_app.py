import os
import joblib
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
model_path = os.path.join(BASE_DIR, "models", "dropout_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "processed", "dropout_ready.csv")

# Load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}. Run Phase 6 first.")

model = joblib.load(model_path)
print("✅ Model loaded successfully.")

# Load sample data
df = pd.read_csv(data_path, sep=';')

X = df.drop(columns=['Target'])
y = df['Target']

# Make predictions
predictions = model.predict(X)
df['Prediction'] = predictions

# Save predictions
output_path = os.path.join(BASE_DIR, "data", "processed", "dropout_predictions.csv")
df.to_csv(output_path, index=False, sep=';')

print("🎉 Phase 9 completed successfully!")
print(f"📁 Predictions saved to: {output_path}")
