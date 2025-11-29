import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (3,126,88,41,235,39.3,0.704,27)

# changing the input data as numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the input data as we are predicting for one data instance only
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# we don't standerdize the input data as the model we imported form the pipeline,
# it will automatically apply scaler to it


# print(std_data)
prediction = loaded_model.predict(input_data_reshaped)
if prediction[0] :
  print("There is high probability that you are diabetic.")
else:
  print("You are most likely safe")
  print("You are Non-Diabetic")
  # print(f'The probability of you being non-diabetic is : {float(testing_data_accuracy):.3f}')