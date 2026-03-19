import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🧠 Mental Health Risk Predictor")

st.markdown("### Enter your details")

# Inputs
age = st.slider("Age", 18, 60)

work_interfere = st.selectbox("Work Stress Level (0–3)", [0,1,2,3])
remote_work = st.selectbox("Remote Work (0=No, 1=Yes)", [0,1])
benefits = st.selectbox("Company Benefits (0/1)", [0,1])
care_options = st.selectbox("Care Options (0/1)", [0,1])
family_history = st.selectbox("Family History (0/1)", [0,1])
mental_health_consequence = st.selectbox("Mental Health Consequence (0/1)", [0,1])

# Prediction
if st.button("Predict"):
    input_data = np.array([[
        age,
        work_interfere,
        remote_work,
        benefits,
        care_options,
        family_history,
        mental_health_consequence
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
        st.write("👉 Improve work-life balance")
        st.write("👉 Seek professional help")
    else:
        st.success("✅ Low Risk")
