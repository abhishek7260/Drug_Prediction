import pickle
import streamlit as st
import sklearn
import numpy as np

# Check scikit-learn version
sklearn_version = sklearn.__version__

# Display scikit-learn version
st.write(f"Using scikit-learn version: {sklearn_version}")

# Load the saved model
try:
    with open('model.pkl', 'rb') as file:
        Drug_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Page title
st.title('Drug Prediction using ML')

# Input fields
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    Age = st.text_input('Age')

with col2:
    Sex = st.selectbox('Sex', ['Female', 'Male'])

with col3:
    BP = st.selectbox('Blood Pressure', ['Low', 'Normal', 'High'])

with col4:
    Cholesterol = st.selectbox('Cholesterol', ['Normal', 'High'])

with col5:
    Na_to_K = st.text_input('Na_to_K')

# Default value for Drug prediction
Drug_prediction = ''

# Prediction button
if st.button('Predict Drug'):
    # Ensure input data is valid and convert to appropriate types
    try:
        # Mapping categorical values to numerical values
        sex_mapping = {'Female': 0, 'Male': 1}
        bp_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
        cholesterol_mapping = {'Normal': 0, 'High': 1}
        
        # Convert input to numerical values based on mappings
        sex_numeric = sex_mapping[Sex]
        bp_numeric = bp_mapping[BP]
        cholesterol_numeric = cholesterol_mapping[Cholesterol]
        
        input_data = np.array([[float(Age), sex_numeric, bp_numeric, cholesterol_numeric, float(Na_to_K)]])
        
        # Predict drug
        Drug_prediction = Drug_model.predict(input_data)
        
        # Mapping numerical prediction to drug name
        drug_mapping = {0: 'Drug A', 1: 'Drug B', 2: 'Drug C', 3: 'Drug X', 4: 'Drug Y'}
        Drug_prediction = drug_mapping[Drug_prediction[0]]
        
        st.success(f'Predicted Drug: {Drug_prediction}')
    except ValueError:
        st.error('Please enter valid numerical values for Age and Na_to_K.')
