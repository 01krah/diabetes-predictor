import streamlit as st
import pandas as pd
import joblib
from sklearn import tree

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Title
st.title("Diabetes Prediction App (Decision Tree Model + Medical Guidelines)")

st.write("""
### Welcome
This app predicts the likelihood of Diabetes based on 3 critical health indicators:
- HbA1c Level
- Blood Glucose Level
- Age

It compares Machine Learning prediction with official Medical Reference guidelines.
""")

# Input fields
hba1c_level = st.number_input("Enter your HbA1c Level (e.g., 5.5)", min_value=0.0, max_value=20.0, step=0.1)
blood_glucose = st.number_input("Enter your Blood Glucose Level (e.g., 120)", min_value=0, max_value=500, step=1)
age = st.number_input("Enter your Age", min_value=0, max_value=120, step=1)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for model prediction
    input_data = pd.DataFrame({
        "HBA1C_LEVEL": [hba1c_level],
        "BLOOD_GLUCOSE_LEVEL": [blood_glucose],
        "AGE": [age]
    })

    # Predict with the trained Decision Tree model
    model_prediction = model.predict(input_data)[0]

    # Predict using official Medical Reference guidelines
    if hba1c_level >= 6.5 or blood_glucose >= 200:
        medical_prediction = "High risk: Diabetic"
    elif (5.7 <= hba1c_level < 6.5) or (140 <= blood_glucose < 200):
        medical_prediction = "Moderate risk: Pre-Diabetic"
    else:
        medical_prediction = "Low risk: Likely Not Diabetic"

    # Display results
    st.subheader("Prediction Results")

    if model_prediction == 1:
        st.success("Model Prediction: High risk (Likely Diabetic)")
    else:
        st.success("Model Prediction: Low risk (Likely Not Diabetic)")

    st.info(f"Medical Reference Prediction: {medical_prediction}")

st.write("---")

# Medical Reference Information
st.subheader("Official Medical Reference Ranges for Diabetes Diagnosis")
st.markdown("""
- HbA1c < 5.7% → Normal  
- HbA1c 5.7% – 6.4% → Pre-Diabetic  
- HbA1c ≥ 6.5% → Diabetic  

- Fasting Blood Glucose < 100 mg/dL → Normal  
- Blood Glucose 100–125 mg/dL → Pre-Diabetic  
- Blood Glucose ≥ 126 mg/dL → Diabetic  
- Post-Meal Blood Glucose ≥ 200 mg/dL → Diabetic  
""")

# Decision Flow Diagram
if st.checkbox("Show Decision Flow Diagram"):
    st.subheader("Decision Flow Based on Trained Model")
    st.code("""
Start
 └── Is HBA1C_LEVEL > 6.70?
     ├── Yes → Predict: Diabetic (class 1)
     └── No
         └── Is BLOOD_GLUCOSE_LEVEL > 210.00?
             ├── Yes → Predict: Diabetic (class 1)
             └── No
                 └── Is AGE > 53.50?
                     ├── No
                     |    └── Is HBA1C_LEVEL > 5.35?
                     |         ├── No → Predict: Not Diabetic (class 0)
                     |         └── Yes
                     |             └── Is AGE > 38.50?
                     |                 ├── No → Predict: Not Diabetic (class 0)
                     |                 └── Yes → Predict: Not Diabetic (class 0)
                     └── Yes
                          └── Is HBA1C_LEVEL > 5.35?
                              ├── No → Predict: Not Diabetic (class 0)
                              └── Yes
                                  └── Is BLOOD_GLUCOSE_LEVEL > 113.00?
                                      ├── No → Predict: Not Diabetic (class 0)
                                      └── Yes → Predict: Not Diabetic (class 0)
""")

# Threshold Comparison - last
if st.checkbox("Show Model vs Medical Thresholds Comparison"):
    st.subheader("Model vs Medical Thresholds Comparison")
    st.markdown("""
    | Feature | Medical Threshold | Model Threshold |
    |:--------|:------------------|:----------------|
    | HbA1c Level | ≥6.5% | >6.7% |
    | Blood Glucose Level | ≥200 mg/dL | >210 mg/dL |
    | Age | Not officially used | Split at 53.5 years |
    """)

    st.write("Our model predicts diabetes at slightly higher HbA1c and Blood Glucose levels compared to standard medical thresholds, and includes Age for further refinement.")
