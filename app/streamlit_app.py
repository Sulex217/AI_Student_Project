import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib


# ----------------------------------------
# Load Model
# ----------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "dropout_model.pkl")
    return joblib.load(model_path)


model = load_model()

# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Manual Prediction", "Batch Prediction", "Admin Dashboard", "About"])

# ----------------------------------------
# Homepage
# ----------------------------------------
if page == "Home":
    st.title("🎓 Student Dropout Prediction System")
    st.markdown(
        """
        Welcome to the Student Dropout Prediction System.  

        Use the sidebar to navigate:
        - **Manual Prediction**: Enter a single student's data.  
        - **Batch Prediction**: Upload a CSV file for multiple students.  
        - **Admin Dashboard**: See statistics and trends.  
        - **About**: Learn how this app works.
        """
    )

# ----------------------------------------
# About Page
# ----------------------------------------
elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    This system predicts the risk of a student dropping out using historical academic and demographic data.  

    **Why features matter:**
    - **Socioeconomic**: Marital status, parental education/occupation, nationality.  
    - **Financial**: Tuition fees, scholarship, debtor.  
    - **Academic**: Grades, enrolled/approved units.  
    - **Macro-economic**: Inflation rate, unemployment rate, GDP.
    """)

# ----------------------------------------
# Manual Single Student Prediction
# ----------------------------------------
elif page == "Manual Prediction":
    st.title("🧍 Manual Single Student Prediction")

    st.sidebar.header("Feature Explanation")
    st.sidebar.markdown(
        "Categorical features are dropdowns or tick boxes; numeric features use sliders or number inputs. Hover over info icons for details.")


    def manual_input():
        data = {}
        # Categorical
        data['Marital status'] = st.selectbox("Marital Status", ["Single", "Married", "Other"],
                                              help="Marital status may affect social responsibilities")
        data['Daytime/evening attendance'] = st.selectbox("Attendance Type", ["Daytime", "Evening"],
                                                          help="Time of day student attends classes")
        data['Previous qualification (grade)'] = st.selectbox("Previous Qualification Grade", ["A", "B", "C", "D"],
                                                              help="Student's prior academic qualification")
        data['Mother\'s qualification'] = st.selectbox("Mother's Qualification",
                                                       ["None", "High School", "Bachelor", "Master", "PhD"],
                                                       help="Mother's education level")
        data['Father\'s qualification'] = st.selectbox("Father's Qualification",
                                                       ["None", "High School", "Bachelor", "Master", "PhD"],
                                                       help="Father's education level")
        data['Mother\'s occupation'] = st.selectbox("Mother's Occupation", ["Unemployed", "Employed", "Self-employed"],
                                                    help="Mother's occupation may affect support")
        data['Father\'s occupation'] = st.selectbox("Father's Occupation", ["Unemployed", "Employed", "Self-employed"],
                                                    help="Father's occupation may affect support")
        data['Educational special needs'] = st.selectbox("Special Needs", ["Yes", "No"],
                                                         help="Students with special needs may require accommodations")
        data['Tuition fees up to date'] = st.selectbox("Tuition Fees Paid", ["Yes", "No"], help="Financial situation")
        data['Scholarship holder'] = st.selectbox("Scholarship Holder", ["Yes", "No"],
                                                  help="Financial support can reduce dropout risk")
        data['International'] = st.selectbox("International Student", ["Yes", "No"],
                                             help="International students face different challenges")
        data['Displaced'] = st.selectbox("Displaced", ["Yes", "No"],
                                         help="Displaced students may have higher dropout risk")
        data['Debtor'] = st.selectbox("Debtor", ["Yes", "No"], help="Students in debt may drop out")
        data['Gender'] = st.selectbox("Gender", ["Male", "Female"], help="Gender can correlate with dropout trends")
        data['Nationality'] = st.selectbox("Nationality", ["Local", "Foreign"],
                                           help="Nationality may affect support networks")
        data['Application mode'] = st.selectbox("Application Mode", ["Regular", "Special"], help="Mode of admission")
        data['Course'] = st.text_input("Course", help="Course enrolled")

        # Numeric
        data['Application order'] = st.number_input("Application Order", min_value=1, max_value=100, value=1)
        data['Admission grade'] = st.number_input("Admission Grade", min_value=0.0, max_value=20.0, value=10.0)
        data['Age at enrollment'] = st.number_input("Age at Enrollment", min_value=15, max_value=40, value=18)
        data['Curricular units 1st sem (enrolled)'] = st.number_input("1st Sem Units Enrolled", min_value=0,
                                                                      max_value=20, value=0)
        data['Curricular units 1st sem (approved)'] = st.number_input("1st Sem Units Approved", min_value=0,
                                                                      max_value=20, value=0)
        data['Curricular units 2nd sem (enrolled)'] = st.number_input("2nd Sem Units Enrolled", min_value=0,
                                                                      max_value=20, value=0)
        data['Curricular units 2nd sem (approved)'] = st.number_input("2nd Sem Units Approved", min_value=0,
                                                                      max_value=20, value=0)
        data['Curricular units 1st sem (grade)'] = st.number_input("1st Sem Average Grade", min_value=0.0,
                                                                   max_value=20.0, value=0.0)
        data['Curricular units 2nd sem (grade)'] = st.number_input("2nd Sem Average Grade", min_value=0.0,
                                                                   max_value=20.0, value=0.0)
        data['Inflation rate'] = st.number_input("Inflation Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
        data['Unemployment rate'] = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
        data['GDP'] = st.number_input("GDP", min_value=0.0, value=5000.0)

        return pd.DataFrame([data])


    input_df = manual_input()

    if st.button("Predict Dropout Risk"):
        prediction = model.predict_proba(input_df)[:, 1][0]
        st.success(f"Predicted dropout risk: {prediction * 100:.2f}%")
        if prediction > 0.5:
            st.warning("⚠️ High risk of dropout")
        else:
            st.info("✅ Low risk of dropout")

# ----------------------------------------
# Batch Prediction (CSV Upload)
# ----------------------------------------
elif page == "Batch Prediction":
    st.title("📥 Batch Prediction (CSV Upload)")
    st.markdown("Upload a CSV file with the same columns as Manual Prediction (excluding target).")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            predictions = model.predict_proba(df)[:, 1]
            df["Dropout Risk (%)"] = predictions * 100
            st.dataframe(df)
            st.success("✅ Batch prediction completed")

            # CSV download
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------------------------------
# Admin Dashboard
# ----------------------------------------
elif page == "Admin Dashboard":
    st.title("📊 Admin Dashboard")
    st.markdown("Summary statistics and model insights.")

    st.subheader("Predicted Dropout Risk Distribution (Sample)")
    sample_preds = np.random.rand(100) * 100
    st.bar_chart(sample_preds)

    st.subheader("Feature Insights")
    st.write("Placeholder: show feature importance, trends, or student demographics here.")

