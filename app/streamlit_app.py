import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎓 Student Dropout Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "models", "dropout_model.pkl")
    return joblib.load(model_path)

model = load_model()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Manual Prediction", "📁 Batch CSV Upload", "🧑‍💼 Admin Dashboard", "ℹ️ About"]
)

# =========================
# SIDEBAR EDUCATION PANEL
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📘 How This System Works")
st.sidebar.write("""
This system predicts whether a student is likely to **drop out** based on:

- Academic performance
- Financial status
- Demographics
- Institutional engagement
- Economic environment

Early identification helps schools:
✔ Provide academic support  
✔ Offer financial aid  
✔ Reduce dropout rates  
✔ Improve student success
""")

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":
    st.title("🎓 Student Dropout Prediction System")
    st.markdown("""
Welcome to the **Student Dropout Prediction System**.

This intelligent system uses machine learning to analyze student data and predict the risk of dropout, enabling institutions to take early, targeted action.

### 🚀 Features
- ✅ Manual single-student prediction
- ✅ Batch prediction using CSV upload
- ✅ Admin dashboard insights
- ✅ Explainable, professional interface

Use the navigation sidebar to get started.
""")

# =========================
# MANUAL PREDICTION PAGE
# =========================
elif page == "📊 Manual Prediction":
    st.title("📊 Manual Single Student Prediction")
    st.write("Fill in the student's details below to predict dropout risk.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            marital_status = st.selectbox(
                "Marital status",
                [0, 1, 2, 3],
                help="Student's marital status: 0=Single, 1=Married, 2=Divorced, 3=Widowed"
            )
            application_mode = st.selectbox(
                "Application mode",
                [1, 2, 3, 4, 5],
                help="How the student applied: 1=General admission, 2=Special admission, 3=Transfer, 4=International, 5=Other"
            )
            application_order = st.number_input(
                "Application order",
                min_value=0, max_value=10, step=1,
                help="Order in which this institution was chosen (1 = first choice, higher numbers = lower preference)"
            )
            course = st.selectbox(
                "Course",
                list(range(1, 20)),
                help="Program of study code. Example: 1=Engineering, 2=Medicine, 3=Law, 4=Education, 5=Business, 6=Science, 7=Arts, 8=Agriculture, others=Institution-specific programs."
            )
            daytime_evening_attendance = st.selectbox(
                "Daytime/Evening attendance",
                [0, 1],
                help="Class schedule: 0=Evening classes, 1=Daytime classes"
            )

        with col2:
            previous_qualification = st.selectbox(
                "Previous qualification",
                list(range(0, 20)),
                help="Highest qualification before admission. Example: 0=None, 1=Primary, 2=Secondary, 3=High school diploma, 4=Vocational, 5=Bachelor’s, others=Higher education levels"
            )
            previous_qualification_grade = st.number_input(
                "Previous qualification (grade)",
                min_value=0.0, max_value=200.0, step=0.1,
                help="Final grade or score obtained in the previous qualification"
            )
            admission_grade = st.number_input(
                "Admission grade",
                min_value=0.0, max_value=200.0, step=0.1,
                help="Grade used by the institution to admit the student"
            )
            nationality = st.selectbox(
                "Nationality",
                list(range(1, 50)),
                help="Student nationality code. Example: 1=Local citizen, 2=Neighboring country, 3=International, others=Country-specific codes"
            )
            displaced = st.checkbox(
                "Displaced",
                help="Tick if the student is displaced due to conflict, disaster, or forced relocation"
            )

        with col3:
            educational_special_needs = st.checkbox(
                "Educational special needs",
                help="Tick if the student has special educational needs or disabilities requiring support"
            )
            debtor = st.checkbox(
                "Debtor",
                help="Tick if the student has outstanding unpaid fees or debts to the institution"
            )
            tuition_fees_up_to_date = st.checkbox(
                "Tuition fees up to date",
                help="Tick if the student has paid all required tuition fees"
            )
            scholarship_holder = st.checkbox(
                "Scholarship holder",
                help="Tick if the student is receiving a scholarship or financial aid"
            )
            gender = st.selectbox(
                "Gender",
                [0, 1],
                help="Student gender: 0=Female, 1=Male"
            )
            age_at_enrollment = st.number_input(
                "Age at enrollment",
                min_value=15, max_value=100, step=1,
                help="Student's age at the time of enrollment"
            )
            international = st.checkbox(
                "International student",
                help="Tick if the student is studying outside their home country"
            )

        st.markdown("### 📚 Academic Performance")

        col4, col5, col6 = st.columns(3)

        with col4:
            cu1_enrolled = st.number_input(
                "Curricular units 1st sem (enrolled)",
                min_value=0, max_value=50, step=1,
                help="Number of courses the student enrolled in during the first semester"
            )
            cu1_evaluations = st.number_input(
                "Curricular units 1st sem (evaluations)",
                min_value=0, max_value=50, step=1,
                help="Number of courses evaluated (with exams/assessments) in the first semester"
            )
            cu1_approved = st.number_input(
                "Curricular units 1st sem (approved)",
                min_value=0, max_value=50, step=1,
                help="Number of courses successfully passed in the first semester"
            )
            cu
