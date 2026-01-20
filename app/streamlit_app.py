import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
st.title("🎓 Student Dropout Prediction System")
model_path = "models/dropout_model.pkl"

try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error(f"❌ Model file not found at {model_path}")

# --- Sidebar instructions ---
st.sidebar.header("How to input data")
st.sidebar.write("""
- Select from dropdowns/tick boxes for categorical fields.
- Use sliders or number inputs for numeric fields.
- Hover over inputs to see tips about what the value means.
- You can also upload a CSV file for batch predictions.
""")

st.header("🧍 Manual Single Student Prediction")

# --- CATEGORICAL INPUTS ---
marital_status = st.selectbox(
    "Marital Status",
    options=["Single", "Married", "Divorced", "Other"],
    help="Select the marital status of the student."
)

application_mode = st.selectbox(
    "Application Mode",
    options=["Standard", "Extraordinary", "Other"],
    help="Select how the student applied."
)

course = st.selectbox(
    "Course",
    options=["Engineering", "Business", "Arts", "Science", "Other"],
    help="Select the course the student is enrolled in."
)

daytime_attendance = st.radio(
    "Daytime/Evening Attendance",
    options=["Daytime", "Evening"],
    help="Select whether the student attends daytime or evening classes."
)

prev_qualification = st.selectbox(
    "Previous Qualification",
    options=["None", "Secondary School", "Bachelor's", "Master's", "PhD"],
    help="Select the highest previous qualification of the student."
)

prev_qualification_grade = st.selectbox(
    "Previous Qualification Grade",
    options=["Poor", "Average", "Good", "Very Good", "Excellent"],
    help="Select the grade achieved in the previous qualification."
)

nationality = st.selectbox(
    "Nationality",
    options=["Domestic", "International"],
    help="Select whether the student is from the home country or abroad."
)

mother_qualification = st.selectbox(
    "Mother's Qualification",
    options=["None", "Secondary School", "Bachelor's", "Master's", "PhD"],
    help="Select the highest qualification of the mother."
)

father_qualification = st.selectbox(
    "Father's Qualification",
    options=["None", "Secondary School", "Bachelor's", "Master's", "PhD"],
    help="Select the highest qualification of the father."
)

mother_occupation = st.selectbox(
    "Mother's Occupation",
    options=["Unemployed", "Blue-collar", "White-collar", "Professional"],
    help="Select the mother's occupation."
)

father_occupation = st.selectbox(
    "Father's Occupation",
    options=["Unemployed", "Blue-collar", "White-collar", "Professional"],
    help="Select the father's occupation."
)

gender = st.radio(
    "Gender",
    options=["Male", "Female", "Other"],
    help="Select the gender of the student."
)

displaced = st.radio(
    "Displaced Student",
    options=["Yes", "No"],
    help="Is the student displaced from another location?"
)

debtor = st.radio(
    "Debtor",
    options=["Yes", "No"],
    help="Does the student have outstanding tuition fees?"
)

scholarship_holder = st.radio(
    "Scholarship Holder",
    options=["Yes", "No"],
    help="Does the student hold a scholarship?"
)

educational_special_needs = st.radio(
    "Educational Special Needs",
    options=["Yes", "No"],
    help="Does the student have special educational needs?"
)

international = st.radio(
    "International",
    options=["Yes", "No"],
    help="Is the student an international student?"
)

# --- NUMERIC INPUTS ---
age_at_enrollment = st.number_input(
    "Age at Enrollment", min_value=15, max_value=60, value=18,
    help="Enter the age of the student at enrollment."
)

admission_grade = st.number_input(
    "Admission Grade", min_value=0.0, max_value=20.0, value=10.0,
    help="Enter the student's admission grade."
)

curr_units_1st_sem_credited = st.number_input(
    "Curricular Units 1st Sem (Credited)", min_value=0, max_value=60, value=5,
    help="Number of curricular units credited in 1st semester."
)
curr_units_1st_sem_enrolled = st.number_input(
    "Curricular Units 1st Sem (Enrolled)", min_value=0, max_value=60, value=5,
    help="Number of curricular units enrolled in 1st semester."
)
curr_units_1st_sem_evaluations = st.number_input(
    "Curricular Units 1st Sem (Evaluations)", min_value=0, max_value=60, value=5,
    help="Number of evaluations in 1st semester."
)
curr_units_1st_sem_approved = st.number_input(
    "Curricular Units 1st Sem (Approved)", min_value=0, max_value=60, value=5,
    help="Number of units approved in 1st semester."
)
curr_units_1st_sem_grade = st.number_input(
    "Curricular Units 1st Sem (Grade)", min_value=0.0, max_value=20.0, value=10.0,
    help="Average grade of 1st semester units."
)
curr_units_1st_sem_without_eval = st.number_input(
    "Curricular Units 1st Sem (Without Evaluations)", min_value=0, max_value=60, value=0,
    help="Number of units without evaluations in 1st semester."
)

curr_units_2nd_sem_credited = st.number_input(
    "Curricular Units 2nd Sem (Credited)", min_value=0, max_value=60, value=5,
    help="Number of curricular units credited in 2nd semester."
)
curr_units_2nd_sem_enrolled = st.number_input(
    "Curricular Units 2nd Sem (Enrolled)", min_value=0, max_value=60, value=5,
    help="Number of curricular units enrolled in 2nd semester."
)
curr_units_2nd_sem_evaluations = st.number_input(
    "Curricular Units 2nd Sem (Evaluations)", min_value=0, max_value=60, value=5,
    help="Number of evaluations in 2nd semester."
)
curr_units_2nd_sem_approved = st.number_input(
    "Curricular Units 2nd Sem (Approved)", min_value=0, max_value=60, value=5,
    help="Number of units approved in 2nd semester."
)
curr_units_2nd_sem_grade = st.number_input(
    "Curricular Units 2nd Sem (Grade)", min_value=0.0, max_value=20.0, value=10.0,
    help="Average grade of 2nd semester units."
)
curr_units_2nd_sem_without_eval = st.number_input(
    "Curricular Units 2nd Sem (Without Evaluations)", min_value=0, max_value=60, value=0,
    help="Number of units without evaluations in 2nd semester."
)

# Economic indicators
unemployment_rate = st.number_input(
    "Unemployment Rate (%)", min_value=0.0, max_value=100.0, value=5.0,
    help="Current unemployment rate in the country."
)
inflation_rate = st.number_input(
    "Inflation Rate (%)", min_value=0.0, max_value=50.0, value=2.0,
    help="Current inflation rate in the country."
)
gdp = st.number_input(
    "GDP (USD billions)", min_value=0.0, max_value=200000.0, value=30000.0, step=100.0,
    help="GDP of the country in USD billions."
)

# --- Prediction Button ---
if st.button("Predict Dropout Risk"):
    # Here you can construct a DataFrame for prediction
    input_data = pd.DataFrame({
        "Marital status": [marital_status],
        "Application mode": [application_mode],
        "Application order": [0],  # Example, can add proper input
        "Course": [course],
        "Daytime/evening attendance": [daytime_attendance],
        "Previous qualification": [prev_qualification],
        "Previous qualification (grade)": [prev_qualification_grade],
        "Nationality": [nationality],
        "Mother's qualification": [mother_qualification],
        "Father's qualification": [father_qualification],
        "Mother's occupation": [mother_occupation],
        "Father's occupation": [father_occupation],
        "Admission grade": [admission_grade],
        "Displaced": [displaced],
        "Educational special needs": [educational_special_needs],
        "Debtor": [debtor],
        "Tuition fees up to date": [0],  # Example
        "Gender": [gender],
        "Scholarship holder": [scholarship_holder],
        "Age at enrollment": [age_at_enrollment],
        "International": [international],
        "Curricular units 1st sem (credited)": [curr_units_1st_sem_credited],
        "Curricular units 1st sem (enrolled)": [curr_units_1st_sem_enrolled],
        "Curricular units 1st sem (evaluations)": [curr_units_1st_sem_evaluations],
        "Curricular units 1st sem (approved)": [curr_units_1st_sem_approved],
        "Curricular units 1st sem (grade)": [curr_units_1st_sem_grade],
        "Curricular units 1st sem (without evaluations)": [curr_units_1st_sem_without_eval],
        "Curricular units 2nd sem (credited)": [curr_units_2nd_sem_credited],
        "Curricular units 2nd sem (enrolled)": [curr_units_2nd_sem_enrolled],
        "Curricular units 2nd sem (evaluations)": [curr_units_2nd_sem_evaluations],
        "Curricular units 2nd sem (approved)": [curr_units_2nd_sem_approved],
        "Curricular units 2nd sem (grade)": [curr_units_2nd_sem_grade],
        "Curricular units 2nd sem (without evaluations)": [curr_units_2nd_sem_without_eval],
        "Unemployment rate": [unemployment_rate],
        "Inflation rate": [inflation_rate],
        "GDP": [gdp],
    })
    st.write("✅ Input data prepared for prediction:")
    st.dataframe(input_data)

    # Prediction example (replace with actual model call)
    # prediction = model.predict(input_data)
    # st.success(f"Predicted Dropout Risk: {prediction[0]}")
    st.info("Prediction logic goes here…")
