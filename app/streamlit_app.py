import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Student Dropout Prediction System", layout="wide")

st.title("🎓 Student Dropout Prediction System")
st.markdown("Upload student data or enter values manually to predict dropout risk.")

# Load model
model_path = os.path.join("models", "dropout_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please make sure 'models/dropout_model.pkl' exists.")
    st.stop()

model = joblib.load(model_path)
st.success("✅ Model loaded successfully.")

# Feature columns (must match training data exactly)
FEATURE_COLUMNS = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Admission grade', 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

# ---------------------------
# Batch Prediction Section
# ---------------------------
st.header("📥 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the same structure (without Target column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in FEATURE_COLUMNS):
        st.error("❌ Uploaded file does not contain the required columns.")
    else:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["Dropout Prediction"] = predictions
        df["Dropout Probability"] = probabilities

        st.success("✅ Predictions completed.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "dropout_predictions.csv", "text/csv")

# ---------------------------
# Manual Prediction Section
# ---------------------------
st.header("🧍 Manual Single Student Prediction")

with st.form("manual_prediction_form"):
    inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            inputs[feature] = st.number_input(feature, value=0.0)

    submitted = st.form_submit_button("🔍 Predict Dropout Risk")

if submitted:
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Dropout (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Dropout (Probability: {probability:.2%})")
