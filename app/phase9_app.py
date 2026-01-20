import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="🎓 Student Dropout Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "dropout_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📥 Upload CSV", "🧍 Manual Prediction", "ℹ️ About"])

# ---------------------------
# Sidebar Explanations
# ---------------------------
st.sidebar.markdown("### ❓ Why These Factors Matter")
st.sidebar.markdown("""
This system predicts whether a student is likely to **drop out**, **graduate**, or remain **enrolled**.

We use:
- **Academic performance** (grades, credits, approvals)
- **Socioeconomic factors** (scholarship, tuition, employment)
- **Demographics** (age, gender, marital status, nationality)
- **Macroeconomic indicators** (GDP, unemployment, inflation)

These factors influence:
- Financial stress
- Academic difficulty
- Institutional support
- External pressures
""")

# ---------------------------
# Feature Definitions
# ---------------------------
categorical_features = {
    "Marital status": {0: "Single", 1: "Married", 2: "Divorced", 3: "Widowed"},
    "Daytime/evening attendance": {0: "Daytime", 1: "Evening"},
    "Educational special needs": {0: "No", 1: "Yes"},
    "Tuition fees up to date": {0: "No", 1: "Yes"},
    "Scholarship holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
    "Displaced": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Gender": {0: "Female", 1: "Male"},
}

numeric_features = [
    "Application order",
    "Previous qualification (grade)",
    "Mother's qualification",
    "Mother's occupation",
    "Admission grade",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
    "Inflation rate",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nationality",
    "Father's qualification",
    "Father's occupation",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "GDP",
]

# ---------------------------
# Home Page
# ---------------------------
if page == "🏠 Home":
    st.title("🎓 Student Dropout Prediction System")
    st.subheader("Predict student dropout risk using AI")

    st.markdown("""
    ### 🔍 What this system does
    This application uses a trained machine learning model to predict:
    - Whether a student will **drop out**
    - **Graduate**
    - Or remain **enrolled**

    ### 🧠 Why this matters
    Early identification of at-risk students allows:
    - Universities to provide **support interventions**
    - Policymakers to improve **education policies**
    - Institutions to reduce **dropout rates**
    """)

# ---------------------------
# Upload CSV Page
# ---------------------------
elif page == "📥 Upload CSV":
    st.title("📥 Batch Prediction (Upload CSV)")

    st.markdown("Upload a CSV file with the **same structure as the training data**, excluding the target column.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("🔮 Predict Dropout Risk"):
            predictions = model.predict(data)
            prediction_probs = model.predict_proba(data)

            data["Prediction"] = predictions
            data["Dropout Risk (%)"] = np.round(prediction_probs[:, 1] * 100, 2)

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            st.download_button(
                label="📥 Download Results",
                data=data.to_csv(index=False),
                file_name="dropout_predictions.csv",
                mime="text/csv",
            )

# ---------------------------
# Manual Prediction Page
# ---------------------------
elif page == "🧍 Manual Prediction":
    st.title("🧍 Manual Single Student Prediction")

    st.markdown("Enter student information below. Tooltips explain each field.")

    input_data = {}

    st.subheader("📌 Categorical Information")
    for feature, options in categorical_features.items():
        label = f"{feature}"
        selected_label = st.radio(
            label,
            list(options.values()),
            help=f"Select the option that best describes the student’s {feature.lower()}."
        )
        selected_value = list(options.keys())[list(options.values()).index(selected_label)]
        input_data[feature] = selected_value

    st.subheader("📊 Numerical Information")
    for feature in numeric_features:
        input_data[feature] = st.number_input(
            feature,
            min_value=0.0,
            step=0.1,
            help=f"Enter the numeric value for {feature.lower()}."
        )

    if st.button("🔮 Predict Dropout Risk"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        st.success(f"🎯 Prediction: **{prediction}**")
        st.info(f"📈 Dropout Risk Probability: **{probability:.2f}%**")

# ---------------------------
# About Page
# ---------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")

    st.markdown("""
    ### 🧠 Model Overview
    This system uses a supervised machine learning model trained on real student data to predict dropout risk.

    ### 📊 Input Data Includes:
    - Academic performance
    - Financial status
    - Demographics
    - Economic indicators

    ### 🌍 Real-World Impact
    - Universities can **identify at-risk students early**
    - Enables **targeted academic support**
    - Helps reduce dropout rates and improve student success

    ### 👨‍🎓 Built By
    This project was developed as part of an AI learning and research initiative by **Sulaiman Dalhatu Halliru**.
    """)
