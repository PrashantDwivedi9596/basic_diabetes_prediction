"""
Created on 29/11/2025
@author: Prashant
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained pipeline model (scaler + classifier)
@st.cache_resource
def load_model():
    return pickle.load(open("trained_model.sav", "rb"))

model = load_model()

# Prediction function
def diabetes_prediction(input_data):
    try:
        input_array = np.asarray(input_data, dtype=float)  # Convert to float safely
    except:
        return "❌ Please enter valid numeric values."

    input_reshaped = input_array.reshape(1, -1)
    prediction = model.predict(input_reshaped)

    if prediction[0] == 1:
        return "⚠️ High probability of Diabetes."
    else:
        return "✅ You are likely Non-Diabetic."

def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="centered")
    st.title("🩺 Diabetes Prediction Web App")
    st.write("Enter patient details below to check diabetes probability")

    # Input fields in columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        BloodPressure = st.text_input("Blood Pressure")
        Insulin = st.text_input("Insulin Level")
        DiabetiesPedigreeFunction = st.text_input("Diabetes Pedigree Function")

    with col2:
        Glucose = st.text_input("Glucose Level")
        SkinThickness = st.text_input("Skin Thickness")
        BMI = st.text_input("BMI")
        Age = st.text_input("Age")

    # Prediction Button
    if st.button("🔍 Get Diabetes Result"):
        input_list = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetiesPedigreeFunction, Age
        ]
        result = diabetes_prediction(input_list)
        st.success(result)

if __name__ == "__main__":
    main()
