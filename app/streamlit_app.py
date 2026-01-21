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
            marital_status = st.selectbox("Marital status", [0, 1, 2, 3], help="0=Single, 1=Married, 2=Divorced, 3=Widowed")
            application_mode = st.selectbox("Application mode", [1, 2, 3, 4, 5], help="Mode of application")
            application_order = st.number_input("Application order", min_value=0, max_value=10, step=1)
            course = st.selectbox("Course", list(range(1, 20)), help="Program/course code")
            daytime_evening_attendance = st.selectbox("Daytime/Evening attendance", [0, 1], help="0=Evening, 1=Daytime")

        with col2:
            previous_qualification = st.selectbox("Previous qualification", list(range(0, 20)), help="Highest previous qualification")
            previous_qualification_grade = st.number_input("Previous qualification (grade)", min_value=0.0, max_value=200.0, step=0.1, help="Grade obtained previously")
            admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, step=0.1, help="Grade used for admission")
            nationality = st.selectbox("Nationality", list(range(1, 50)), help="Nationality code")
            displaced = st.checkbox("Displaced", help="Tick if displaced")

        with col3:
            educational_special_needs = st.checkbox("Educational special needs", help="Tick if special needs")
            debtor = st.checkbox("Debtor", help="Tick if student has unpaid debts")
            tuition_fees_up_to_date = st.checkbox("Tuition fees up to date", help="Tick if paid all tuition")
            scholarship_holder = st.checkbox("Scholarship holder", help="Tick if receiving scholarship")
            gender = st.selectbox("Gender", [0, 1], help="0=Female, 1=Male")
            age_at_enrollment = st.number_input("Age at enrollment", min_value=15, max_value=100, step=1)
            international = st.checkbox("International student", help="Tick if studying abroad")

        st.markdown("### 📚 Academic Performance")
        col4, col5, col6 = st.columns(3)

        with col4:
            cu1_enrolled = st.number_input("CU 1st sem (enrolled)", min_value=0, max_value=50, step=1)
            cu1_evaluations = st.number_input("CU 1st sem (evaluations)", min_value=0, max_value=50, step=1)
            cu1_approved = st.number_input("CU 1st sem (approved)", min_value=0, max_value=50, step=1)
            cu1_credited = st.number_input("CU 1st sem (credited)", min_value=0, max_value=50, step=1)
            cu1_without_eval = st.number_input("CU 1st sem (without eval)", min_value=0, max_value=50, step=1)
            cu1_grade = st.number_input("CU 1st sem (grade)", min_value=0.0, max_value=200.0, step=0.1)

        with col5:
            cu2_enrolled = st.number_input("CU 2nd sem (enrolled)", min_value=0, max_value=50, step=1)
            cu2_evaluations = st.number_input("CU 2nd sem (evaluations)", min_value=0, max_value=50, step=1)
            cu2_approved = st.number_input("CU 2nd sem (approved)", min_value=0, max_value=50, step=1)
            cu2_credited = st.number_input("CU 2nd sem (credited)", min_value=0, max_value=50, step=1)
            cu2_without_eval = st.number_input("CU 2nd sem (without eval)", min_value=0, max_value=50, step=1)
            cu2_grade = st.number_input("CU 2nd sem (grade)", min_value=0.0, max_value=200.0, step=0.1)

        with col6:
            mother_qualification = st.selectbox("Mother's qualification", list(range(0, 10)), help="Mother's highest qualification")
            mother_occupation = st.selectbox("Mother's occupation", list(range(0, 20)), help="Mother's occupation code")
            father_qualification = st.selectbox("Father's qualification", list(range(0, 10)), help="Father's highest qualification")
            father_occupation = st.selectbox("Father's occupation", list(range(0, 20)), help="Father's occupation code")
            inflation_rate = st.number_input("Inflation rate", min_value=-10.0, max_value=50.0, step=0.1)
            unemployment_rate = st.number_input("Unemployment rate", min_value=0.0, max_value=50.0, step=0.1)
            gdp = st.number_input("GDP", min_value=0.0, max_value=100000.0, step=10.0)

        # ✅ FIXED SUBMIT BUTTON
        submitted = st.form_submit_button("🔍 Predict Dropout Risk")

        if submitted:
            input_data = [
                marital_status, application_mode, application_order, course, daytime_evening_attendance,
                previous_qualification, previous_qualification_grade, nationality, displaced,
                educational_special_needs, debtor, tuition_fees_up_to_date, scholarship_holder,
                gender, age_at_enrollment, international, admission_grade,
                mother_qualification, mother_occupation, father_qualification, father_occupation,
                inflation_rate, unemployment_rate, gdp,
                cu1_enrolled, cu1_evaluations, cu1_approved, cu1_credited, cu1_without_eval, cu1_grade,
                cu2_enrolled, cu2_evaluations, cu2_approved, cu2_credited, cu2_without_eval, cu2_grade
            ]

            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict_proba(input_array)[0][1]

            st.markdown("---")
            st.subheader("📈 Prediction Result")
            st.metric("Dropout Risk Probability", f"{prediction*100:.2f}%")

            if prediction >= 0.5:
                st.error("⚠️ High risk of dropout. Immediate intervention recommended.")
            else:
                st.success("✅ Low risk of dropout. Student is likely to continue.")

# =========================
# BATCH CSV UPLOAD PAGE
# =========================
elif page == "📁 Batch CSV Upload":
    st.title("📁 Batch Prediction via CSV Upload")
    st.write("Upload a CSV file with the same structure as the training data (without target column).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())
        if st.button("🚀 Run Batch Prediction"):
            probabilities = model.predict_proba(df.to_numpy())[:,1]
            df["Dropout_Risk_Probability"] = probabilities
            st.subheader("📊 Prediction Results")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Results CSV", csv, "dropout_predictions.csv", "text/csv")

# =========================
# ADMIN DASHBOARD PAGE
# =========================
elif page == "🧑‍💼 Admin Dashboard":
    st.title("🧑‍💼 Admin Dashboard")
    st.write("This dashboard provides insights into student dropout risk patterns.")
    uploaded_file = st.file_uploader("Upload dataset for dashboard analysis", type=["csv"], key="admin")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Dataset Overview")
        st.write(df.describe())
        if "Dropout_Risk_Probability" in df.columns:
            st.subheader("📈 Risk Distribution")
            st.bar_chart(df["Dropout_Risk_Probability"])
        st.subheader("🧠 Key Insights")
        st.write("""
        - High dropout risk often correlates with:
          - Low academic performance
          - Financial difficulties
          - Lack of parental education
          - Poor engagement (few approved courses)
        - Early identification allows:
          - Academic counseling
          - Financial aid allocation
          - Psychological and social support
        """)

# =========================
# ABOUT PAGE
# =========================
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("""
### 🎓 Student Dropout Prediction System

This application uses machine learning to predict student dropout risk using historical academic, demographic, financial, and institutional data.

#### 🔍 Why This Matters
Student dropout is a major challenge worldwide, leading to:
- Lost educational opportunities
- Financial loss for institutions
- Reduced workforce development

#### 🧠 How It Works
A trained machine learning model analyzes:
- Academic performance trends
- Socioeconomic background
- Institutional engagement
- Economic conditions

The output is a **probability score** indicating dropout risk, enabling proactive intervention.

#### 👨‍💻 Developed by
Sulaiman Halliru Dalhatu   
AI Student Project — 2026
""")
