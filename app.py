import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import time

# Debug: Show current directory contents
st.write("Current directory files:", os.listdir('.'))

# Robust file loading with multiple attempts
def load_model():
    model = None
    scaler = None
    
    # Try different possible file paths
    model_paths = [
        'diabetes_model.pkl',
        './diabetes_model.pkl',
        '/mount/src/your-repo-name/diabetes_model.pkl'  # Replace with your actual repo name
    ]
    
    scaler_paths = [
        'scaler.pkl', 
        './scaler.pkl',
        '/mount/src/your-repo-name/scaler.pkl'  # Replace with your actual repo name
    ]
    
    for model_path in model_paths:
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            st.success(f"Model loaded from: {model_path}")
            break
        except FileNotFoundError:
            continue
    
    for scaler_path in scaler_paths:
        try:
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            st.success(f"Scaler loaded from: {scaler_path}")
            break
        except FileNotFoundError:
            continue
    
    return model, scaler

# Load model and scaler
model, scaler = load_model()

if model is None or scaler is None:
    st.error("""
    Model files missing! Please ensure:
    1. diabetes_model.pkl and scaler.pkl are in the app directory
    2. Files are properly committed to GitHub
    3. File names match exactly (case-sensitive)
    """)
    st.stop()

# Your existing app code continues here...
st.title("Diabetes Prediction App")

# Add your input fields and prediction logic here
# ... rest of your app code
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Health AI - Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body {font-family:'Inter',sans-serif;background:linear-gradient(135deg,#f0fdfa,#ecfdf5,#f9fafb);}
        .main {background-color:#ffffffcc;padding:2rem 3rem;border-radius:18px;box-shadow:0 4px 25px rgba(0,0,0,0.05);backdrop-filter:blur(6px);}
        .title{text-align:center;font-size:42px;font-weight:800;background:linear-gradient(90deg,#0f766e,#14b8a6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2em;}
        .subtitle{text-align:center;color:#0d9488;font-size:18px;margin-bottom:40px;opacity:0.8;}
        .stButton>button{width:100%;border:none;background:linear-gradient(90deg,#0d9488,#14b8a6);color:white;
                         font-size:18px;font-weight:600;border-radius:10px;padding:0.6rem 1.5rem;
                         box-shadow:0 2px 8px rgba(13,148,136,0.3);transition:all 0.2s ease;}
        .stButton>button:hover{background:linear-gradient(90deg,#115e59,#0d9488);transform:translateY(-1.5px);
                               box-shadow:0 4px 12px rgba(13,148,136,0.4);}
        .info-card{border-radius:14px;padding:1.2rem 1.5rem;margin-top:1.5rem;color:#065f46;animation:fadeIn 0.4s ease-in;}
        @keyframes fadeIn{from{opacity:0;transform:translateY(5px);}to{opacity:1;transform:translateY(0);}}
        .metric-value{font-size:28px;font-weight:700;color:#0f766e;}
        footer{visibility:hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='title'>🩺 Health AI - Diabetes Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered risk prediction for a healthier future</p>", unsafe_allow_html=True)
st.divider()

# --- INPUT SECTION ---
st.markdown("### 🧬 Enter Your Health Data")
col1, col2, col3 = st.columns(3)
with col1:
    pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1)
    glucose = st.number_input("🍬 Glucose Level", 0, 300, 120)
    bp = st.number_input("💉 Blood Pressure", 0, 200, 70)
with col2:
    skin_thickness = st.number_input("🧪 Skin Thickness", 0, 100, 20)
    insulin = st.number_input("💊 Insulin Level", 0, 900, 80)
    bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0)
with col3:
    dpf = st.number_input("🩸 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("🎂 Age", 1, 120, 25)

st.write("")

# --- PREDICTION SECTION ---
if st.button("🔍 Analyze My Risk"):
    data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)

    with st.spinner('Analyzing your health data...'):
        time.sleep(0.6)
        prediction = model.predict(data_scaled)
        prob = model.predict_proba(data_scaled)[0][1]

    st.write("")
    if prediction[0] == 1:
        st.markdown(
            f"""
            <div class='info-card' style='background-color:#fef2f2;border-left:6px solid #dc2626;'>
                <h3>🚨 High Risk Detected</h3>
                <p>You have a <b class='metric-value'>{prob*100:.1f}%</b> chance of diabetes.</p>
                <p>We recommend consulting a healthcare provider soon.</p>
                <p>Adopt a balanced diet, stay active, and monitor glucose regularly.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='info-card' style='background-color:#f0fdf4;border-left:6px solid #16a34a;'>
                <h3>✅ Low Risk</h3>
                <p>Your estimated chance of diabetes is <b class='metric-value'>{prob*100:.1f}%</b>.</p>
                <p>Maintain your healthy lifestyle — regular checkups are still important.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- DYNAMIC WELLNESS TIPS ---
    st.divider()
    st.markdown("### 🌿 Personalized Health Recommendations")

    if prob < 0.3:
        st.success("🩺 Excellent! Keep up your routine.")
        st.markdown("""
        - 🏃‍♂️ Stay active for at least 150 minutes/week.  
        - 🥗 Eat whole grains, fruits, and lean proteins.  
        - 💧 Keep yourself hydrated.  
        - 😴 Ensure 7–8 hours of quality sleep.  
        """)
    elif 0.3 <= prob < 0.6:
        st.warning("⚠️ Moderate Risk — Early care can make a big difference.")
        st.markdown("""
        - 🏃‍♀️ Include daily walks or light jogging.  
        - 🥦 Reduce sugar and processed foods.  
        - 🧘 Try yoga or meditation to manage stress.  
        - 🔬 Schedule a preventive health check-up in the next few months.  
        """)
    else:
        st.error("🚨 High Risk — Please take proactive action.")
        st.markdown("""
        - 🩺 Consult a doctor for diagnostic testing.  
        - 🍲 Follow a diabetes-friendly diet with low carbs and high fiber.  
        - 🧍 Monitor blood glucose and maintain a healthy weight.  
        - 🚶‍♂️ Engage in 30–45 mins of physical activity daily.  
        - 💊 Follow any prescribed medication and track your progress.  
        """)

    st.markdown("<br><center style='color:#6b7280;'>© 2025 Health AI | Designed with precision and care 🩺</center>", unsafe_allow_html=True)
