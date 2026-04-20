
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from job_titles import JOB_TITLE_OPTIONS # Import the job titles

# Load the best model and column order
best_model = joblib.load('best_salary_prediction_model.pkl')
X_train_columns = joblib.load('X_train_columns.pkl')

st.title('Salary Prediction App')
st.write('Enter employee details to predict their salary.')

# Input widgets for user features
age = st.slider('Age', 18, 65, 30)
years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0)

gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
education_level = st.selectbox('Education Level', ['Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'High School'])

# Use the imported job titles
job_title = st.selectbox('Job Title', JOB_TITLE_OPTIONS)

if st.button('Predict Salary'):
    # Create a DataFrame for the new input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Years of Experience': [years_of_experience],
        'Gender': [gender],
        'Education Level': [education_level],
        'Job Title': [job_title]
    })

    # One-hot encode categorical features (mimic preprocessing during training)
    # Create a template DataFrame with all possible one-hot encoded columns from X_train_columns
    # This ensures consistency in column order and presence
    processed_input = pd.DataFrame(0, index=[0], columns=X_train_columns)

    # Fill in numerical features
    processed_input['Age'] = age
    processed_input['Years of Experience'] = years_of_experience

    # Fill in one-hot encoded categorical features
    gender_col = f'Gender_{gender}'
    if gender_col in processed_input.columns: # Check if column exists, as 'Other' might not always be present if not in training data
        processed_input[gender_col] = 1

    education_col = f'Education Level_{education_level}'
    if education_col in processed_input.columns:
        processed_input[education_col] = 1

    job_title_col = f'Job Title_{job_title}'
    if job_title_col in processed_input.columns:
        processed_input[job_title_col] = 1

    # Make prediction
    predicted_salary = best_model.predict(processed_input)[0]

    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
