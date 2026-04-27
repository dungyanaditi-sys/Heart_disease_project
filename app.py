import streamlit as st
import numpy as np
import pickle

# -----------------------
# Load model & scaler
# -----------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    st.error("❌ Model files not found. Make sure model.pkl and scaler.pkl are present.")
    st.stop()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# -----------------------
# Inputs
# -----------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", step=0.1)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

# Convert categorical
sex = 1 if sex == "Male" else 0

# -----------------------
# Prediction
# -----------------------
if st.button("Predict"):

    try:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ High chance of Heart Disease")
        else:
            st.success("✅ Low chance of Heart Disease")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("About")
st.sidebar.write(
    "This app uses Machine Learning (Random Forest) to predict heart disease risk."
)