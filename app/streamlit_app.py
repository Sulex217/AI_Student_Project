import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Dropout Prediction System", layout="wide")

st.title("🎓 Student Dropout Prediction System")

# ------------------------------
# Load model
# ------------------------------
model_path = "app/models/dropout_model.pkl"
try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# ------------------------------
# Instructions
# ------------------------------
st.markdown("""
This app predicts student dropout risk.

**Manual Single Student Prediction:** Enter values for a student in each field.  
Hover over each field label to see what the values mean.  

**CSV Batch Prediction:** Upload a CSV file with the same columns (excluding the Target column).
""")

# ------------------------------
# Sample tooltips dictionary
# ------------------------------
tooltips = {
    "Marital status": "0: Single, 1: Married, 2: Other",
    "Application mode": "0: Online, 1: In-person, 2: Other",
    "Application order": "Numeric order of application",
    "Course": "Course code or name",
    "Daytime/evening attendance": "0: Daytime, 1: Evening",
    "Previous qualification": "0: None, 1: High School, 2: Bachelor, etc.",
    "Previous qualification (grade)": "Numeric grade",
    "Nationality": "0: Home, 1: International",
    "Mother's qualification": "0: None, 1: Primary, 2: Secondary, 3: Higher",
    "Father's qualification": "0: None, 1: Primary, 2: Secondary, 3: Higher",
    "Mother's occupation": "Categorical code for occupation",
    "Father's occupation": "Categorical code for occupation",
    "Admission grade": "Numeric grade",
    "Displaced": "0: No, 1: Yes",
    "Educational special needs": "0: No, 1: Yes",
    "Debtor": "0: No, 1: Yes",
    "Tuition fees up to date": "0: No, 1: Yes",
    "Gender": "0: Female, 1: Male",
    "Scholarship holder": "0: No, 1: Yes",
    "Age at enrollment": "Numeric age",
    "International": "0: No, 1: Yes",
    "Curricular units 1st sem (credited)": "Numeric value",
    "Curricular units 1st sem (enrolled)": "Numeric value",
    "Curricular units 1st sem (evaluations)": "Numeric value",
    "Curricular units 1st sem (approved)": "Numeric value",
    "Curricular units 1st sem (grade)": "Numeric grade",
    "Curricular units 1st sem (without evaluations)": "Numeric value",
    "Curricular units 2nd sem (credited)": "Numeric value",
    "Curricular units 2nd sem (enrolled)": "Numeric value",
    "Curricular units 2nd sem (evaluations)": "Numeric value",
    "Curricular units 2nd sem (approved)": "Numeric value",
    "Curricular units 2nd sem (grade)": "Numeric grade",
    "Curricular units 2nd sem (without evaluations)": "Numeric value",
    "Unemployment rate": "Percentage, e.g., 5.3",
    "Inflation rate": "Percentage, e.g., 2.1",
    "GDP": "Numeric value"
}

# ------------------------------
# Manual single student prediction
# ------------------------------
st.header("🧍 Manual Single Student Prediction")
manual_input = {}
for col, tooltip in tooltips.items():
    manual_input[col] = st.number_input(f"{col}", help=tooltip, value=0.0)

if st.button("Predict Dropout Risk"):
    try:
        input_df = pd.DataFrame([manual_input])
        prediction = model.predict_proba(input_df)[:, 1][0]  # probability of dropout
        st.success(f"Predicted Dropout Risk: {prediction*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------
# CSV batch prediction
# ------------------------------
st.header("📥 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV (without Target column)", type="csv")

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("Predict Dropout Risk for Batch"):
            predictions = model.predict_proba(batch_df)[:, 1]
            batch_df["Predicted Dropout Risk"] = predictions
            st.success("✅ Batch predictions completed.")
            st.dataframe(batch_df)
            st.download_button(
                "Download Predictions CSV",
                batch_df.to_csv(index=False).encode('utf-8'),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
