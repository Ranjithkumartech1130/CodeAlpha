import streamlit as st
import numpy as np
import pickle
import os

# Set page configuration for a premium look
st.set_page_config(
    page_title="Credit Scoring Intelligence",
    page_icon="💳",
    layout="centered",
)

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return pickle.load(open("model.pkl", "rb"))
    return None


model = load_model()

# Header Section
st.title("💳 Credit Scoring Model")
st.markdown("### Intelligent Financial Assessment")
st.write("Enter the applicant's financial details below to predict creditworthiness.")

if model is None:
    st.error("⚠️ Model file (model.pkl) not found. Please run 'train_model.py' first.")
else:
    # Form layout
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            income = st.number_input(
                "Annual Income ($)", min_value=0, value=50000, step=1000)
            debt = st.number_input(
                "Total Debt ($)", min_value=0, value=10000, step=500)
            payment_history = st.selectbox("Payment History", options=[(
                "Good", 1), ("Bad", 0)], format_func=lambda x: x[0])

        with col2:
            credit_utilization = st.slider(
                "Credit Utilization Ratio", 0.0, 1.0, 0.3, 0.01)
            credit_years = st.number_input(
                "Credit History (Years)", min_value=0, value=5, step=1)

    st.markdown("---")

    # Prediction logic
    if st.button("Analyze Creditworthiness"):
        features = np.array(
            [[income, debt, payment_history[1], credit_utilization, credit_years]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.markdown(
                '<div class="prediction-box success">Result: Creditworthy ✅</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(
                '<div class="prediction-box danger">Result: Not Creditworthy ❌</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed by 🩷Dhanyashree Ranjith Kumar💜")
