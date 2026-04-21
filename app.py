import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💰 Salary Prediction App")

st.write("Enter details below to predict salary")

# Inputs
rating = st.number_input("Company Rating", 0.0, 5.0, 3.0)
salary_reported = st.number_input("Salaries Reported", 1, 1000, 10)

# Predict button
if st.button("Predict Salary"):
    input_data = np.array([[rating, salary_reported]])
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    
    st.success(f"Predicted Salary: ₹ {int(prediction[0])}")
