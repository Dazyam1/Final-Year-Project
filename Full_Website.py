import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# Utility function to load models safely
def load_model_safe(file_path, model_type='joblib'):
    try:
        if model_type == 'joblib':
            return joblib.load(file_path)
        elif model_type == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load models and vectorizer
hepatitis_model = load_model_safe('hepatitis_model.pkl', 'pickle')
tb_model = load_model_safe('tb_predictor_model.pkl', 'joblib')
hiv_model = load_model_safe('hiv_model.pkl', 'joblib')
vectorizer = load_model_safe('vectorizer.pkl', 'joblib')

# App title
st.title("Medical Diagnosis Prediction App")
st.write("Predict **Hepatitis**, **Tuberculosis**, or **HIV** based on input data.")

# Sidebar for selecting disease
app_mode = st.sidebar.selectbox("Choose Prediction Task", ["Hepatitis Prediction", "Tuberculosis Prediction", "HIV Prediction"])

# Hepatitis Prediction
if app_mode == "Hepatitis Prediction":
    st.header("Hepatitis Prediction")
    st.subheader("Enter Patient Information")

    age = st.slider("Age", 0, 100, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    malaise = st.selectbox("Malaise", ["Yes", "No"])
    anorexia = st.selectbox("Anorexia", ["Yes", "No"])
    liver_big = st.selectbox("Liver Enlargement", ["Yes", "No"])
    liver_firm = st.selectbox("Liver Firmness", ["Yes", "No"])
    spleen_palpable = st.selectbox("Spleen Palpable", ["Yes", "No"])
    spiders = st.selectbox("Spider Angiomas", ["Yes", "No"])
    ascites = st.selectbox("Ascites", ["Yes", "No"])

    if st.button("Predict Hepatitis"):
        sex_binary = 1 if sex == "Male" else 2
        convert = lambda x: 1 if x == "Yes" else 2

        features = np.array([
            age,
            sex_binary,
            convert(fatigue),
            convert(malaise),
            convert(anorexia),
            convert(liver_big),
            convert(liver_firm),
            convert(spleen_palpable),
            convert(spiders),
            convert(ascites)
        ]).reshape(1, -1)

        if hepatitis_model:
            prediction = hepatitis_model.predict(features)[0]
            st.success("Hepatitis Detected" if prediction == 1 else "No Hepatitis Detected")

# Tuberculosis Prediction
elif app_mode == "Tuberculosis Prediction":
    st.header("Tuberculosis Prediction")
    st.subheader("Enter Patient Information")

    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    duration = st.number_input("Duration (days)", min_value=0, value=10)
    cough = st.selectbox("Cough", ["Yes", "No"])
    weight_loss = st.selectbox("Weight Loss", ["Yes", "No"])
    fever = st.selectbox("Fever", ["Yes", "No"])
    night_sweats = st.selectbox("Night Sweats", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])

    if st.button("Predict Tuberculosis"):
        to_bin = lambda x: 1 if x == "Yes" else 0
        features = np.array([
            age, duration,
            to_bin(cough),
            to_bin(weight_loss),
            to_bin(fever),
            to_bin(night_sweats),
            to_bin(chest_pain),
            to_bin(fatigue)
        ]).reshape(1, -1)

        if tb_model:
            prediction = tb_model.predict(features)[0]
            st.success("Tuberculosis Detected" if prediction == 1 else "No Tuberculosis Detected")

# HIV Prediction
elif app_mode == "HIV Prediction":
    st.header("HIV Chat-based Symptom Prediction")
    st.write("Describe your symptoms or situation in a sentence (e.g. _'I have lost weight and feel constantly tired.'_):")

    user_input = st.text_area("Your message")

    if st.button("Predict HIV"):
        if vectorizer and hiv_model and user_input.strip():
            transformed_input = vectorizer.transform([user_input])
            prediction = hiv_model.predict(transformed_input)[0]
            st.success("HIV Detected" if prediction == 1 else "No HIV Detected")
        elif not user_input.strip():
            st.warning("Please enter a symptom description.")
