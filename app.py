import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("🧠 Mental Health Risk Prediction System")

st.markdown("""
This AI system predicts mental health risk based on:
- Workplace stress
- Company support
- Personal factors

👉 Helps in early detection and prevention.
""")

# Input Section
st.subheader("📋 Enter your details")

age = st.slider("Age", 18, 60)

work_interfere = st.selectbox("Work Stress Level (0–3)", [0,1,2,3])
remote_work = st.selectbox("Remote Work (0=No, 1=Yes)", [0,1])
benefits = st.selectbox("Company Benefits (0/1)", [0,1])
care_options = st.selectbox("Care Options (0/1)", [0,1])
family_history = st.selectbox("Family History (0/1)", [0,1])
mental_health_consequence = st.selectbox("Mental Health Consequence (0/1)", [0,1])

# Prediction
if st.button("Predict Risk"):
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

    # Probability prediction
    prob = model.predict_proba(input_scaled)[0][1]

    # Result
    st.subheader("📊 Result")

    if prob < 0.4:
        st.success("✅ Low Mental Health Risk")
    elif prob < 0.7:
        st.warning("⚠️ Medium Mental Health Risk")
    else:
        st.error("🚨 High Mental Health Risk")

    st.write(f"🔍 Confidence Score: {prob:.2f}")

    # Recommendations
    st.subheader("💡 Recommendations")

    if prob > 0.7:
        st.write("👉 Seek professional help")
        st.write("👉 Talk to HR or counselor")
        st.write("👉 Reduce workload stress")

    elif prob > 0.4:
        st.write("👉 Take regular breaks")
        st.write("👉 Improve work-life balance")

    else:
        st.write("👉 Maintain healthy routine")
        st.write("👉 Continue good work-life balance")
        st.write("👉 Stay socially connected")

    # 🔥 Dynamic Chart (FIXED)
    st.subheader("📈 Mental Health Factors Impact")

    stress_impact = work_interfere * 30
    support_impact = (benefits + care_options) * 25
    family_impact = family_history * 40

    chart_data = {
        "Factor": ["Stress", "Support", "Family History"],
        "Impact": [stress_impact, support_impact, family_impact]
    }

    st.bar_chart(chart_data)
