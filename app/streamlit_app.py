# app/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Insurance Premium Estimator", layout="centered")

st.title("🏥 Insurance Premium Predictor")

with st.form("input_form"):
    age = st.slider("Age", 18, 100)
    height = st.number_input("Height (cm)", 100, 220)
    weight = st.number_input("Weight (kg)", 30, 200)
    diabetes = st.checkbox("Diabetes")
    bp_problems = st.checkbox("Blood Pressure Problems")
    transplants = st.checkbox("Any Transplants")
    chronic = st.checkbox("Any Chronic Diseases")
    allergies = st.checkbox("Known Allergies")
    cancer_history = st.checkbox("History Of Cancer In Family")
    major_surgeries = st.selectbox("Number Of Major Surgeries", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Premium")

if submitted:
    payload = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Diabetes": diabetes,
        "BloodPressureProblems": bp_problems,
        "AnyTransplants": transplants,
        "AnyChronicDiseases": chronic,
        "KnownAllergies": allergies,
        "HistoryOfCancerInFamily": cancer_history,
        "NumberOfMajorSurgeries": major_surgeries
    }

    response = requests.post("http://localhost:5001/predict", json=payload)
    
    if response.status_code == 200:
        st.success(f"💰 Estimated Premium: ₹ {response.json()['predicted_premium']}")
    else:
        st.error("❌ Prediction failed. Please check inputs or server.")
