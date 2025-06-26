
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
from tensorflow import keras

df = pd.read_csv('diabetes.csv')

df.head()

#Checking the no. of rows and column
df.shape

#Basic info about the dataset
df.info()

#Dividing the dataset into dependent and independent variable
X = df.drop(columns ='Outcome')
Y = df['Outcome']

X.head()

#Normlize the feature
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Save the scaler
joblib.dump(scaler, 'Scaler.pkl')

#Dividing teh dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X.shape[1]

#Build Data Analysis
model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(X.shape[1],), activation='relu'), #imput layer
    keras.layers.Dense(8, activation='relu'), #hidden layer
    keras.layers.Dense(1, activation='sigmoid') #output layer
])

#printing the summary of model
model.summary()

#complie the model
model.compile(optimizer='adam', loss='binary_crossentropy')

#train the model
model.fit(X_train, Y_train, epochs=50, batch_size =10, validation_data=(X_test, Y_test))

#making the preediction and conerting it into int vari
Y_predict= model.predict(X_test)
Y_predict = (Y_predict > 0.5).astype("int32")

#calculate the performancr matrix
accuracy = accuracy_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
cm = confusion_matrix(Y_test, Y_predict)
cr = classification_report(Y_test, Y_predict)

#print all the performance matrices
print("The Accuracy of our diabetics prediction model is: ", accuracy)
print("The Recall of our diabetics prediction model is: ", recall)
print("The Precision of our diabetics prediction model is: ", precision)
print("The F1 Score of our diabetics prediction model is: ", f1)
print("The Confusion Matrix of our diabetics prediction model is: \n", cm)
print("The Classification Report of our diabetics prediction model is: \n", cr)

model.save("Diabetes_model.h5")
print("The model has been saved")


import streamlit as st

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

