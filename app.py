import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the encoders and scaler
try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")

# Streamlit App
st.title('Customer Churn Prediction')
st.markdown("### Predict the likelihood of a customer leaving the business")

# Collapsible section for user inputs
with st.expander("Enter Customer Details"):
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0.0, format="%0.2f")
    credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, step=1)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%0.2f")
    tenure = st.slider('Tenure (years)', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', ["No", "Yes"])
    is_active_member = st.selectbox('Is Active Member', ["No", "Yes"])

# Prepare input data
try:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active_member == "Yes" else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine input data with encoded geography
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # # Predict churn
    prediction = model.predict(input_data_scaled)
    # prediction_proba = prediction[0][0]

    # Convert prediction_proba to a Python float
    prediction_proba = float(prediction[0][0])

    # Display the progress bar
    st.progress(prediction_proba)

    # Display results
    st.markdown("### Prediction Results")
    st.progress(prediction_proba)

   

    st.write(f'Churn Probability: {prediction_proba:.2%}')

    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
except Exception as e:
    st.error(f"Error during prediction: {e}")
