import joblib
import streamlit as st
from tensorflow import keras

from keras.models import load_model
model = load_model("Diabetes_model.h5", compile=False)


model = keras.models.load_model("Diabetes_model.h5")
scaler = joblib.load("Scaler.pkl")

st.set_page_config(page_title='Diabetes Prediction App', layout='centered')
st.title('Diabetes Prediction App')
st.markdown("Enter the following datails to predict the diabetes")

#Input fields
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=10, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=0)
bloodpressure = st.number_input('Blood Pressure', min_value=0, max_value=0)
skinthickness = st.number_input('Skin Thickness', min_value=0, max_value=0) 
insulin = st.number_input("Enter the amount of Insulin Level in Patient body", min_value=0, max_value=0)
bmi = st.number_input('BMI', min_value=1)
diabetespedigreefunction = st.number_input('Diabetes Pedigree Function', min_value=0)
age = st.number_input('Age', min_value=0)

#make prediction 
if st.button('Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "Not Diabetic" if prediction < 0.5 else "Diabetic"
    st.subheader("the result of the prediction are:", result)