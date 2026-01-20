import streamlit as st
import pandas as pd
import joblib

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="🎓 Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ================================
# Load Model
# ================================
@st.cache_resource
def load_model():
    return joblib.load("model/student_dropout_model.pkl")

model = load_model()

# ================================
# Feature Names (Corrected)
# ================================
FEATURES = [
    "Marital status",
    "Application order",
    "Daytime/evening attendance",
    "Previous qualification (grade)",
    "Mother's qualification",
    "Mother's occupation",
    "Admission grade",
    "Educational special needs",
    "Tuition fees up to date",
    "Scholarship holder",
    "International",
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
    "Displaced",
    "Debtor",
    "Gender",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "GDP"
]

# ================================
# Sidebar Navigation
# ================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Batch Prediction (CSV)", "🧍 Manual Prediction", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.info("""
### ℹ️ How to Use This App
- Use **Batch Prediction** to upload CSV files.
- Use **Manual Prediction** to input a single student.
- Hover on each input for explanations.
- Sidebar explains why each feature matters.
""")

# ================================
# Sidebar Explanations
# ================================
st.sidebar.markdown("### 📘 Why These Factors Matter")

st.sidebar.write("""
This system predicts **student dropout risk** using:
- Academic performance
- Financial stability
- Family background
- Institutional and economic factors

Even non-academic features (like GDP or marital status) influence stress, focus, and persistence — all proven predictors of dropout.
""")

# ================================
# HOME PAGE
# ================================
if page == "🏠 Home":
    st.title("🎓 Student Dropout Prediction System")
    st.markdown("""
    Welcome to the **Student Dropout Prediction System**.

    This application uses machine learning to help:
    - Universities
    - Academic advisors
    - Policymakers
    - Students

    identify students who are **at risk of dropping out** — early enough to intervene.

    ---
    ### 🚀 What You Can Do:
    - Upload a CSV file to predict dropout risk for many students.
    - Manually input a student's data to get an instant prediction.
    - Understand why each factor matters through tooltips and sidebar explanations.
    """)

    st.success("✅ Model loaded successfully.")

# ================================
# BATCH PREDICTION PAGE
# ================================
elif page == "📊 Batch Prediction (CSV)":
    st.title("📊 Batch Prediction via CSV Upload")

    st.markdown("""
    Upload a CSV file containing student data (without the target column).
    The system will return dropout predictions for all records.
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("📄 Uploaded Data Preview:")
        st.dataframe(data.head())

        if st.button("🔍 Predict Dropout Risk"):
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            data["Dropout Prediction"] = predictions
            data["Dropout Risk Probability"] = probabilities

            st.success("✅ Predictions completed.")
            st.dataframe(data.head())

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                "dropout_predictions.csv",
                "text/csv"
            )

# ================================
# MANUAL PREDICTION PAGE
# ================================
elif page == "🧍 Manual Prediction":
    st.title("🧍 Manual Single Student Prediction")

    st.markdown("""
    Enter the student's details below.  
    Categorical features use **tick boxes / select boxes**.  
    Numeric features use **sliders or number inputs**.  
    Hover over each input for explanations.
    """)

    input_data = {}

    # ================================
    # Categorical Inputs
    # ================================
    st.subheader("📌 Categorical Information")

    input_data["Marital status"] = st.selectbox(
        "Marital status",
        options=[0, 1, 2, 3, 4],
        help="0: Single, 1: Married, 2: Divorced, 3: Widowed, 4: Other"
    )

    input_data["Daytime/evening attendance"] = st.radio(
        "Attendance type",
        options=[1, 0],
        format_func=lambda x: "Daytime" if x == 1 else "Evening",
        help="1: Daytime, 0: Evening classes"
    )

    input_data["Educational special needs"] = st.radio(
        "Educational special needs",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1: Has special educational needs"
    )

    input_data["Tuition fees up to date"] = st.radio(
        "Tuition fees up to date",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="1: Fees are paid up, 0: Fees are not paid"
    )

    input_data["Scholarship holder"] = st.radio(
        "Scholarship holder",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1: Student has a scholarship"
    )

    input_data["International"] = st.radio(
        "International student",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1: Student is an international student"
    )

    input_data["Displaced"] = st.radio(
        "Displaced",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1: Student is displaced from original residence"
    )

    input_data["Debtor"] = st.radio(
        "Debtor",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1: Student has outstanding debt"
    )

    input_data["Gender"] = st.radio(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="0: Female, 1: Male"
    )

    # ================================
    # Numeric Inputs
    # ================================
    st.subheader("📊 Academic & Economic Information")

    input_data["Application order"] = st.number_input(
        "Application order",
        min_value=0,
        max_value=10,
        value=1,
        help="Order in which the student applied (1 = first choice)"
    )

    input_data["Previous qualification (grade)"] = st.slider(
        "Previous qualification (grade)",
        min_value=0.0,
        max_value=200.0,
        value=120.0,
        help="Student's grade from previous qualification"
    )

    input_data["Mother's qualification"] = st.number_input(
        "Mother's qualification (coded)",
        min_value=0,
        max_value=20,
        value=10,
        help="Education level of mother (coded value)"
    )

    input_data["Mother's occupation"] = st.number_input(
        "Mother's occupation (coded)",
        min_value=0,
        max_value=20,
        value=5,
        help="Occupation category of mother (coded value)"
    )

    input_data["Admission grade"] = st.slider(
        "Admission grade",
        min_value=0.0,
        max_value=200.0,
        value=130.0,
        help="Grade at admission into institution"
    )

    input_data["Curricular units 1st sem (enrolled)"] = st.number_input(
        "Curricular units 1st sem (enrolled)",
        min_value=0,
        max_value=20,
        value=6,
        help="Number of courses enrolled in first semester"
    )

    input_data["Curricular units 1st sem (approved)"] = st.number_input(
        "Curricular units 1st sem (approved)",
        min_value=0,
        max_value=20,
        value=5,
        help="Number of courses passed in first semester"
    )

    input_data["Curricular units 1st sem (without evaluations)"] = st.number_input(
        "Curricular units 1st sem (without evaluations)",
        min_value=0,
        max_value=20,
        value=0,
        help="Courses without evaluation in first semester"
    )

    input_data["Curricular units 2nd sem (enrolled)"] = st.number_input(
        "Curricular units 2nd sem (enrolled)",
        min_value=0,
        max_value=20,
        value=6,
        help="Number of courses enrolled in second semester"
    )

    input_data["Curricular units 2nd sem (approved)"] = st.number_input(
        "Curricular units 2nd sem (approved)",
        min_value=0,
        max_value=20,
        value=5,
        help="Number of courses passed in second semester"
    )

    input_data["Curricular units 2nd sem (without evaluations)"] = st.number_input(
        "Curricular units 2nd sem (without evaluations)",
        min_value=0,
        max_value=20,
        value=0,
        help="Courses without evaluation in second semester"
    )

    input_data["Inflation rate"] = st.slider(
        "Inflation rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        help="National inflation rate at enrollment time"
    )

    input_data["Application mode"] = st.number_input(
        "Application mode (coded)",
        min_value=0,
        max_value=20,
        value=1,
        help="Type of application used (coded value)"
    )

    input_data["Course"] = st.number_input(
        "Course (coded)",
        min_value=0,
        max_value=200,
        value=50,
        help="Course identifier code"
    )

    input_data["Previous qualification"] = st.number_input(
        "Previous qualification (coded)",
        min_value=0,
        max_value=20,
        value=1,
        help="Type of previous qualification (coded value)"
    )

    input_data["Nationality"] = st.number_input(
        "Nationality (coded)",
        min_value=0,
        max_value=200,
        value=1,
        help="Nationality code"
    )

    input_data["Father's qualification"] = st.number_input(
        "Father's qualification (coded)",
        min_value=0,
        max_value=20,
        value=10,
        help="Education level of father (coded value)"
    )

    input_data["Father's occupation"] = st.number_input(
        "Father's occupation (coded)",
        min_value=0,
        max_value=20,
        value=5,
        help="Occupation category of father (coded value)"
    )

    input_data["Age at enrollment"] = st.slider(
        "Age at enrollment",
        min_value=15,
        max_value=60,
        value=20,
        help="Student's age when enrolling"
    )

    input_data["Curricular units 1st sem (credited)"] = st.number_input(
        "Curricular units 1st sem (credited)",
        min_value=0,
        max_value=20,
        value=0,
        help="Courses credited in first semester"
    )

    input_data["Curricular units 1st sem (evaluations)"] = st.number_input(
        "Curricular units 1st sem (evaluations)",
        min_value=0,
        max_value=50,
        value=10,
        help="Total evaluations in first semester"
    )

    input_data["Curricular units 1st sem (grade)"] = st.slider(
        "Curricular units 1st sem (grade)",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        help="Average grade in first semester"
    )

    input_data["Curricular units 2nd sem (credited)"] = st.number_input(
        "Curricular units 2nd sem (credited)",
        min_value=0,
        max_value=20,
        value=0,
        help="Courses credited in second semester"
    )

    input_data["Curricular units 2nd sem (evaluations)"] = st.number_input(
        "Curricular units 2nd sem (evaluations)",
        min_value=0,
        max_value=50,
        value=10,
        help="Total evaluations in second semester"
    )

    input_data["Curricular units 2nd sem (grade)"] = st.slider(
        "Curricular units 2nd sem (grade)",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        help="Average grade in second semester"
    )

    input_data["Unemployment rate"] = st.slider(
        "Unemployment rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        help="National unemployment rate at enrollment time"
    )

    input_data["GDP"] = st.slider(
        "GDP (index value)",
        min_value=0.0,
        max_value=100000.0,
        value=50000.0,
        help="Gross Domestic Product indicator"
    )

    # ================================
    # Prediction Button
    # ================================
    if st.button("🔍 Predict Dropout Risk"):
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Dropout Risk Detected! (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Low Dropout Risk (Probability: {probability:.2%})")

        st.markdown("### 📊 Input Summary")
        st.dataframe(df)

# ================================
# ABOUT PAGE
# ================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")

    st.markdown("""
    ### 🎓 Student Dropout Prediction System

    This system was developed to:
    - Predict which students are at risk of dropping out.
    - Support early academic interventions.
    - Improve student retention and success.

    ### 🤖 How It Works
    A machine learning model was trained using:
    - Academic records
    - Socio-economic indicators
    - Family background
    - Institutional and national data

    The model learns complex patterns that human judgment alone might miss.

    ### 🌍 Real-World Impact
    - Universities can identify at-risk students early.
    - Advisors can offer targeted support.
    - Governments can improve education policy.
    - Students receive help before failure occurs.

    ### 🛠️ Technologies Used
    - Python
    - Scikit-learn
    - Pandas
    - Streamlit

    ### 👨‍💻 Developed by
    **Sulaiman Dalhatu Halliru**
    """)
