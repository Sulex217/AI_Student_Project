import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Student Dropout Prediction System",
    layout="wide"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = os.path.join("models", "dropout_model.pkl")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully.")
else:
    st.error(f"❌ Model not found at {MODEL_PATH}. Run Phase 6 first.")

# ----------------------------
# TAB INTERFACE
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Batch Prediction", "Manual Single Prediction", "Feature Importance / Trends"])

# ----------------------------
# BATCH PREDICTION
# ----------------------------
with tab1:
    st.subheader("📥 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV with student data (without Target column)", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file, sep=';')
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())
            
            # Make predictions
            predictions = model.predict_proba(df_batch)[:,1]
            df_batch["Dropout Probability"] = predictions
            df_batch["Risk Level"] = pd.cut(predictions, bins=[-1,0.33,0.66,1], labels=["Low","Medium","High"])
            
            st.success("Predictions generated!")
            st.dataframe(df_batch.head())
            
            # Download CSV
            csv_file = df_batch.to_csv(index=False, sep=';')
            st.download_button("Download Predictions CSV", csv_file, file_name="batch_predictions.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------------------
# MANUAL SINGLE STUDENT PREDICTION
# ----------------------------
with tab2:
    st.subheader("🧍 Manual Single Student Prediction")
    
    # Define input columns (example based on your CSV)
    input_data = {}
    columns = [
        'Marital status', 'Application mode', 'Application order', 'Course',
        'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)',
        'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
        'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    for col in columns:
        input_data[col] = st.number_input(label=col, value=0.0)
    
    if st.button("Predict Dropout Risk"):
        input_df = pd.DataFrame([input_data])
        probability = model.predict_proba(input_df)[:,1][0]
        if probability < 0.33:
            risk = "Low"
            color = "green"
        elif probability < 0.66:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"
        
        st.markdown(f"**Dropout Probability:** {probability:.2f}")
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>Risk Level: {risk}</span>", unsafe_allow_html=True)

# ----------------------------
# FEATURE IMPORTANCE / TRENDS
# ----------------------------
with tab3:
    st.subheader("📊 Feature Importance / Trends")
    
    try:
        importances = model.feature_importances_
        feat_names = columns
        feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        st.write("Top Features contributing to Dropout:")
        st.bar_chart(feat_df.set_index('Feature')['Importance'])
        
        # Optional: Risk distribution
        if 'df_batch' in locals():
            st.write("Dropout Risk Distribution (Batch Prediction):")
            st.bar_chart(df_batch['Risk Level'].value_counts())
        
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

