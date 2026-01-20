import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# App Configuration
# ==============================
st.set_page_config(
    page_title="🎓 Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout Prediction System")
st.markdown("""
Predict whether a student is at risk of dropping out using academic, demographic, 
and socio-economic information.
""")

# ==============================
# Load Model
# ==============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dropout_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==============================
# Feature List (MUST match training data order)
# ==============================
FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📥 Batch Prediction", "🧍 Single Student Prediction", "ℹ️ About the System"])

# ==============================
# Home Page
# ==============================
if page == "🏠 Home":
    st.header("Welcome 👋")
    st.markdown("""
    This system predicts **student dropout risk** using machine learning.

    ### 🔍 What you can do:
    - Upload a CSV file for **batch predictions**
    - Enter student data manually for **single predictions**
    - Understand how the system works and how to use it effectively

    ### 🎯 Who should use this?
    - Universities & colleges
    - Academic advisors
    - Educational policymakers
    - Student support services
    """)

# ==============================
# Batch Prediction Page
# ==============================
elif page == "📥 Batch Prediction":
    st.header("📥 Batch Prediction (CSV Upload)")

    st.markdown("""
    Upload a CSV file **with the same structure as the training dataset**, 
    but **without the `Target` column**.
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully.")
            st.dataframe(df.head())

            missing_cols = [col for col in FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                if st.button("🔮 Run Batch Prediction"):
                    predictions = model.predict(df[FEATURES])
                    probabilities = model.predict_proba(df[FEATURES])[:, 1]

                    df["Dropout Prediction"] = predictions
                    df["Dropout Risk Probability"] = probabilities

                    st.success("✅ Predictions completed.")
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        csv,
                        "dropout_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ==============================
# Single Student Prediction Page
# ==============================
elif page == "🧍 Single Student Prediction":
    st.header("🧍 Single Student Prediction")

    st.markdown("""
    Enter the student's information below. All values must be **numeric** 
    and follow the same encoding as the training dataset.
    """)

    col1, col2 = st.columns(2)
    inputs = {}

    for i, feature in enumerate(FEATURES):
        if i % 2 == 0:
            with col1:
                inputs[feature] = st.number_input(feature, value=0.0, format="%.2f")
        else:
            with col2:
                inputs[feature] = st.number_input(feature, value=0.0, format="%.2f")

    if st.button("🔮 Predict Dropout Risk"):
        try:
            input_df = pd.DataFrame([inputs])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error(f"⚠️ Student is **likely to drop out** (Risk: {probability:.2%})")
            else:
                st.success(f"✅ Student is **likely to continue** (Risk: {probability:.2%})")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ==============================
# About Page
# ==============================
elif page == "ℹ️ About the System":
    st.header("ℹ️ About This System")

    st.markdown("""
    ### 🧠 Model Used
    This application uses a **machine learning classification model** trained on student 
    academic, demographic, and socio-economic data to predict dropout risk.

    ### ⚙️ How It Works
    - The model analyzes patterns in student performance and background.
    - It outputs:
        - A **binary prediction** (Dropout / Continue)
        - A **probability score** indicating risk level.

    ### 🌍 Real-World Impact
    This system helps:
    - Identify at-risk students early.
    - Enable targeted interventions.
    - Improve student retention and graduation rates.
    - Support data-driven educational decisions.

    ### 📈 Ethical Use
    Predictions should be used to **support students**, not punish them.
    Always combine model outputs with human judgment and institutional context.
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("© 2026 Student Dropout Prediction System | Built with Streamlit & Machine Learning")
