import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model
model = joblib.load('../models/copd_model.pkl')

# Title for the Streamlit app
st.title('COPD Risk Prediction')

# Sidebar inputs for patient details
age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
smoking_status = st.sidebar.selectbox('Smoking Status', ['Former', 'Never', 'Current'])
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
biomass_exposure = st.sidebar.selectbox('Biomass Fuel Exposure', [0, 1])
occupational_exposure = st.sidebar.selectbox('Occupational Exposure', [0, 1])
family_history_copd = st.sidebar.selectbox('Family History of COPD', [0, 1])
location = st.sidebar.selectbox('Location', ['Lalitpur', 'Pokhara', 'Kathmandu'])
air_pollution_level = st.sidebar.number_input('Air Pollution Level', min_value=0, max_value=500, value=100)
respiratory_infections_childhood = st.sidebar.selectbox('Respiratory Infections in Childhood', [0, 1])

# Preprocess input data
def preprocess_input(age, gender, smoking_status, bmi, biomass_exposure, occupational_exposure, family_history_copd, location, air_pollution_level, respiratory_infections_childhood):
    # Encoding and scaling input
    gender_encoded = 1 if gender == 'Male' else 0
    smoking_status_encoded = [1, 0] if smoking_status == 'Former' else [0, 1] if smoking_status == 'Never' else [0, 0]
    location_encoded = [1, 0] if location == 'Kathmandu' else [0, 1] if location == 'Pokhara' else [0, 0]
    
    scaled_age = (age - 44.84) / 13.77
    scaled_bmi = (bmi - 25.98) / 4.20
    scaled_air_pollution_level = (air_pollution_level - 140.93) / 67.42
    
    return np.array([scaled_age, gender_encoded, *smoking_status_encoded, biomass_exposure, occupational_exposure, family_history_copd, scaled_bmi, *location_encoded, scaled_air_pollution_level, respiratory_infections_childhood]).reshape(1, -1)

# Preprocess the input
input_data = preprocess_input(age, gender, smoking_status, bmi, biomass_exposure, occupational_exposure, family_history_copd, location, air_pollution_level, respiratory_infections_childhood)

# Predict COPD risk
prediction = model.predict(input_data)

# Display result
if st.button('Predict'):
    if prediction == 1:
        st.write("The patient is at risk of COPD.")
    else:
        st.write("The patient is not at risk of COPD.")
