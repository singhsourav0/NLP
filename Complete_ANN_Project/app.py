import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)


st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on their information.")
st.write("Please enter the customer information below:")


# Input fields for customer information
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=100, value=30)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=600)
tenure = st.number_input("Tenure (in years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=2000000.0, value=50000.0)


# Convert input data to DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    'Geography' : [geography],
    'EstimatedSalary': [estimated_salary]
    
})


geography_encoded = one_hot_encoder.transform(input_data[["Geography"]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder.get_feature_names_out(["Geography"]))


input_data = pd.concat([input_data.drop('Geography', axis=1), geography_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f"Prediction: The customer is likely to churn (Probability: {prediction_proba:.2f})")
else:
    st.write(f"Prediction: The customer is likely to stay (Probability: {1 - prediction_proba:.2f})")