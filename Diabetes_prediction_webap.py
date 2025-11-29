"""
created on 29/11/2025
@auther: Prashant
"""
import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))



# creating the function for the prediction
def diabeties_prediction(input_data):
      # input_data = (3,126,88,41,235,39.3,0.704,27)
      # changing the input data as numpy array
      input_data_as_numpy_array = np.asarray(input_data)

      # reshaping the input data as we are predicting for one data instance only
      input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
      # we don't standerdize the input data as the model we imported form the pipeline,
      # it will automatically apply scaler to it

      prediction = loaded_model.predict(input_data_reshaped)
      if prediction[0] :
        return "There is high probability that you are diabetic."
      else:
        return "You are most likely safe \nYou are Non-Diabetic"
def main():
    # giving a title
    st.title('Diabeties Prediction Webapp')

    # getting the input data from the users
    Pregnancies =st.text_input('Number of Pregnancies : ')
    Glucose = st.text_input('The Glucose level : ')
    BloodPressure = st.text_input('The value of Blood Pressure : ')
    SkinThickness = st.text_input('SkinThickness : ')
    Insulin = st.text_input('Insulin Levels : ')
    BMI = st.text_input("Value of BMI : ")
    DiabetiesPedigreeFunction = st.text_input('Diebeties Pedigree Function value : ')
    Age = st.text_input("Your age : ")

    # code for the prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabeties Test Result'):
        diagnosis =  diabeties_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI, DiabetiesPedigreeFunction,Age])
    st.success(diagnosis)
if __name__ == "__main__":
    main()