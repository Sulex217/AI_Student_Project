# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- Paths ---
MODEL_PATH = os.path.join("models", "dropout_model.pkl")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- Sidebar ---
st.sidebar.title("Student Dropout Prediction System")
st.sidebar.markdown("""
**Welcome!** This app predicts the risk of student dropout.

**Feature Guide:**
- **Marital Status**: Could indicate family responsibilities affecting study time.
- **Admission Grade**: Prior academic performance.
- **Curricular Units**: Completed credits can show progress.
- **Economic Indicators (GDP, Inflation)**: Provide context for financial stress risks.
- **Scholarship Holder, Tuition Fees up to Date**: Financial stability.
""")

# --- Navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["Home", "CSV Upload", "Manual Prediction", "About"]
)

# --- Home Page ---
if page == "Home":
    st.title("🎓 Student Dropout Prediction System")
    st.markdown("""
    Welcome! This system predicts student dropout risk. You can:
    - Upload a CSV file for batch predictions
    - Enter a single student’s information manually
    Use the sidebar for guidance on what each input means.
    """)

# --- CSV Upload Page ---
elif page == "CSV Upload":
    st.title("📥 Batch Prediction (CSV Upload)")
    st.markdown("Upload a CSV file with the same structure (exclude Target column).")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("Predict Dropout Risk"):
                predictions = model.predict(df)
                df["Dropout Risk"] = predictions
                st.success("Predictions complete!")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- Manual Single Student Prediction ---
elif page == "Manual Prediction":
    st.title("🧍 Manual Single Student Prediction")
    st.markdown("Enter student information manually:")

    with st.form("manual_prediction_form"):
        # Categorical Inputs
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Other"],
            help="Social/family responsibilities affecting study time."
        )

        application_mode = st.selectbox(
            "Application Mode", ["Regular", "Extraordinary"],
            help="Admission mode can affect student preparedness."
        )

        course = st.selectbox(
            "Course", ["Engineering", "Science", "Business", "Arts"],
            help="Student's course of study."
        )

        nationality = st.selectbox(
            "Nationality", ["Domestic", "International"],
            help="Domestic or international students may have different dropout risks."
        )

        previous_qualification = st.selectbox(
            "Previous Qualification", ["High School", "Associate Degree", "Other"],
            help="Prior academic qualifications."
        )

        mother_qualification = st.selectbox(
            "Mother's Qualification", ["None", "High School", "College", "Other"],
            help="Parental education can influence support."
        )

        father_qualification = st.selectbox(
            "Father's Qualification", ["None", "High School", "College", "Other"],
            help="Parental education can influence support."
        )

        mother_occupation = st.selectbox(
            "Mother's Occupation", ["Unemployed", "Employed", "Other"],
            help="Parental occupation may affect financial stability."
        )

        father_occupation = st.selectbox(
            "Father's Occupation", ["Unemployed", "Employed", "Other"],
            help="Parental occupation may affect financial stability."
        )

        gender = st.selectbox(
            "Gender", ["Male", "Female"],
            help="Student gender."
        )

        displaced = st.selectbox(
            "Displaced", ["Yes", "No"],
            help="Whether the student has been displaced."
        )

        debtor = st.selectbox(
            "Debtor", ["Yes", "No"],
            help="Student has outstanding debts."
        )

        tuition_fees_up_to_date = st.selectbox(
            "Tuition Fees Up to Date", ["Yes", "No"],
            help="Financial obligations can affect dropout risk."
        )

        scholarship_holder = st.selectbox(
            "Scholarship Holder", ["Yes", "No"],
            help="Scholarship support can reduce dropout risk."
        )

        international = st.selectbox(
            "International", ["Yes", "No"],
            help="International students may have different support networks."
        )

        educational_special_needs = st.selectbox(
            "Educational Special Needs", ["Yes", "No"],
            help="Special needs may require additional support."
        )

        # Numeric Inputs
        age_at_enrollment = st.number_input(
            "Age at Enrollment", min_value=15, max_value=50, value=18,
            help="Student age at enrollment."
        )

        admission_grade = st.slider(
            "Admission Grade", min_value=0.0, max_value=20.0, value=10.0, step=0.1,
            help="Admission grade shows prior academic performance."
        )

        curricular_units_1st_sem_credited = st.number_input(
            "Curricular Units 1st Sem (Credited)", min_value=0.0, value=0.0,
            help="Credits successfully completed in first semester."
        )

        curricular_units_1st_sem_enrolled = st.number_input(
            "Curricular Units 1st Sem (Enrolled)", min_value=0.0, value=0.0,
            help="Credits the student is enrolled for in first semester."
        )

        curricular_units_1st_sem_evaluations = st.number_input(
            "Curricular Units 1st Sem (Evaluations)", min_value=0.0, value=0.0
        )

        curricular_units_1st_sem_approved = st.number_input(
            "Curricular Units 1st Sem (Approved)", min_value=0.0, value=0.0
        )

        curricular_units_1st_sem_grade = st.number_input(
            "Curricular Units 1st Sem (Grade)", min_value=0.0, max_value=20.0, value=0.0
        )

        curricular_units_1st_sem_without_evaluations = st.number_input(
            "Curricular Units 1st Sem (Without Evaluations)", min_value=0.0, value=0.0
        )

        curricular_units_2nd_sem_credited = st.number_input(
            "Curricular Units 2nd Sem (Credited)", min_value=0.0, value=0.0
        )

        curricular_units_2nd_sem_enrolled = st.number_input(
            "Curricular Units 2nd Sem (Enrolled)", min_value=0.0, value=0.0
        )

        curricular_units_2nd_sem_evaluations = st.number_input(
            "Curricular Units 2nd Sem (Evaluations)", min_value=0.0, value=0.0
        )

        curricular_units_2nd_sem_approved = st.number_input(
            "Curricular Units 2nd Sem (Approved)", min_value=0.0, value=0.0
        )

        curricular_units_2nd_sem_grade = st.number_input(
            "Curricular Units 2nd Sem (Grade)", min_value=0.0, max_value=20.0, value=0.0
        )

        curricular_units_2nd_sem_without_evaluations = st.number_input(
            "Curricular Units 2nd Sem (Without Evaluations)", min_value=0.0, value=0.0
        )

        unemployment_rate = st.number_input(
            "Unemployment Rate", min_value=0.0, max_value=100.0, value=0.0,
            help="Local unemployment rate in percentage."
        )

        inflation_rate = st.number_input(
            "Inflation Rate", min_value=0.0, max_value=100.0, value=0.0,
            help="Local inflation rate in percentage."
        )

        gdp = st.number_input(
            "GDP", min_value=0.0, value=0.0,
            help="Local Gross Domestic Product."
        )

        submitted = st.form_submit_button("Predict Dropout Risk")
        if submitted:
            input_data = pd.DataFrame([{
                "Marital status": marital_status,
                "Application mode": application_mode,
                "Course": course,
                "Nationality": nationality,
                "Previous qualification": previous_qualification,
                "Mother's qualification": mother_qualification,
                "Father's qualification": father_qualification,
                "Mother's occupation": mother_occupation,
                "Father's occupation": father_occupation,
                "Gender": gender,
                "Displaced": displaced,
                "Debtor": debtor,
                "Tuition fees up to date": tuition_fees_up_to_date,
                "Scholarship holder": scholarship_holder,
                "International": international,
                "Educational special needs": educational_special_needs,
                "Age at enrollment": age_at_enrollment,
                "Admission grade": admission_grade,
                "Curricular units 1st sem (credited)": curricular_units_1st_sem_credited,
                "Curricular units 1st sem (enrolled)": curricular_units_1st_sem_enrolled,
                "Curricular units 1st sem (evaluations)": curricular_units_1st_sem_evaluations,
                "Curricular units 1st sem (approved)": curricular_units_1st_sem_approved,
                "Curricular units 1st sem (grade)": curricular_units_1st_sem_grade,
                "Curricular units 1st sem (without evaluations)": curricular_units_1st_sem_without_evaluations,
                "Curricular units 2nd sem (credited)": curricular_units_2nd_sem_credited,
                "Curricular units 2nd sem (enrolled)": curricular_units_2nd_sem_enrolled,
                "Curricular units 2nd sem (evaluations)": curricular_units_2nd_sem_evaluations,
                "Curricular units 2nd sem (approved)": curricular_units_2nd_sem_approved,
                "Curricular units 2nd sem (grade)": curricular_units_2nd_sem_grade,
                "Curricular units 2nd sem (without evaluations)": curricular_units_2nd_sem_without_evaluations,
                "Unemployment rate": unemployment_rate,
                "Inflation rate": inflation_rate,
                "GDP": gdp
            }])
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Dropout Risk: {prediction}")

# --- About Page ---
elif page == "About":
    st.title("ℹ️ About the Model")
    st.markdown("""
    **Model**: Random Forest Classifier (example)  
    **Purpose**: Predict student dropout risk  
    **Usage**: Use CSV for batch prediction or manual input for single students.  
    **Real-world impact**: Helps universities and schools identify at-risk students and offer support early.
    """)
