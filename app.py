import streamlit as st
import pickle
import numpy as np
import os

# ----------------------------
# Load model safely
# ----------------------------
model_path = os.path.join(os.getcwd(), "model.pkl")
scaler_path = os.path.join(os.getcwd(), "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health Risk Prediction System")

st.markdown("""
This AI model predicts whether a person is at **risk of mental health issues**  
based on workplace and personal factors.
""")

st.markdown("### 📝 Enter your details")

# ----------------------------
# Inputs
# ----------------------------
age = st.slider("Age", 18, 60)

work_interfere = st.selectbox(
    "Work Stress Level",
    [0, 1, 2, 3],
    help="0=Never, 1=Rarely, 2=Sometimes, 3=Often"
)

remote_work = st.selectbox("Remote Work", [0, 1], help="0=No, 1=Yes")

benefits = st.selectbox("Company Benefits", [0, 1])

care_options = st.selectbox("Care Options Available", [0, 1])

family_history = st.selectbox("Family History of Mental Illness", [0, 1])

mental_health_consequence = st.selectbox(
    "Mental Health Consequence at Work",
    [0, 1]
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Risk"):

    input_data = np.array([[
        age,
        work_interfere,
        remote_work,
        benefits,
        care_options,
        family_history,
        mental_health_consequence
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # ----------------------------
    # Output
    # ----------------------------
    st.markdown("### 📊 Result")

    if prediction[0] == 1:
        st.error("⚠️ High Mental Health Risk")

        st.markdown("""
### 💡 Recommendations:
- Improve work-life balance  
- Take regular breaks  
- Seek professional help  
- Use company wellness programs  
        """)

    else:
        st.success("✅ Low Mental Health Risk")

        st.markdown("""
### 👍 Suggestions:
- Maintain healthy routine  
- Continue good work-life balance  
- Stay socially connected  
        """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed using Machine Learning + Streamlit 🚀")
