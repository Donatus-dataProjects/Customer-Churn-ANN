import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the trained model and pickled objects
#try:
    #model = tf.keras.models.load_model('model.h5')
#except Exception as e:
    #st.error(f"Error loading model: {e}")

# Loading the trained model and pickled objects
try:
    # Attempt to load the model
    model = tf.keras.models.load_model('model.h5')
    if model is None:
        raise ValueError("Model could not be loaded.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    
# Loading pickled files (encoders and scaler)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Building the Streamlit app for customer churn prediction
st.title('Customer Churn Prediction')

# User inputs
credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0] if len(onehot_encoder_geo.categories_) > 0 else [])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance', min_value=0.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})



# One-hot encode the 'Geography' column and use .toarray() to convert sparse matrix to dense
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# Concatenating the 'geo_encoded_df' with input_data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Scaling the input_data
input_data_scaled = scaler.transform(input_data)

# Display loading spinner and make prediction
with st.spinner('Making prediction...'):
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        st.write(f'Churn Probability: {prediction_proba:.2f}')

        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
    except Exception as e:
        st.error(f"Error making prediction: {e}") 

