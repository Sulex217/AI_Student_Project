import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Student Dropout Prediction App", layout="centered")

# App title
st.title("🎓 Student Dropout Prediction System")
st.write("Upload student data or enter values manually to predict dropout risk.")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(BASE_DIR, "models", "dropout_model.pkl")
if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Please run Phase 6 first.")
    st.stop()

model = joblib.load(model_path)
st.success("✅ Model loaded successfully.")

# Load processed dataset structure
data_path = os.path.join(BASE_DIR, "data", "processed", "dropout_ready.csv")
df_structure = pd.read_csv(data_path, sep=';')

# Features
feature_columns = df_structure.drop(columns=["Target"]).columns.tolist()

st.header("📥 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the same structure (without Target column)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file, sep=';')

    missing_cols = set(feature_columns) - set(input_df.columns)
    if missing_cols:
        st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
    else:
        predictions = model.predict(input_df[feature_columns])
        input_df["Prediction"] = predictions

        st.success("🎉 Predictions generated!")
        st.dataframe(input_df.head())

        # Download button
        csv = input_df.to_csv(index=False, sep=';').encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="dropout_predictions.csv",
            mime="text/csv"
        )

st.header("🧍 Manual Single Student Prediction")

with st.form("manual_input_form"):
    user_input = {}

    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0.0)

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ This student is at **HIGH risk of dropout**.")
    else:
        st.success("✅ This student is **NOT likely to dropout**.")
