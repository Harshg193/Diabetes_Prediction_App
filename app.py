import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import os

# Streamlit UI Config
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("Enter the following details to predict diabetes:")

# Load model and scaler safely
try:
    model = tf.keras.models.load_model('Diabetes_model.h5', compile=False)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

try:
    scaler = joblib.load('Scaler.pkl')
except Exception as e:
    st.error(f"Failed to load the scaler: {e}")
    st.stop()

# Input fields
pregnancies = st.number_input("Number of Pregnancies:", min_value=0, max_value=10, value=1)
glucose = st.number_input("Glucose Level:", min_value=0)
blood_pressure = st.number_input("Blood Pressure:", min_value=0)
skin_thickness = st.number_input("Skin Thickness:", min_value=0)
insulin = st.number_input("Insulin Level:", min_value=0)
bmi = st.number_input("BMI:", min_value=1.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function:", min_value=0.0, step=0.01)
age = st.number_input("Age:", min_value=0)

# Predict button
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = "Not Diabetic" if prediction[0][0] < 0.5 else "Diabetic"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
