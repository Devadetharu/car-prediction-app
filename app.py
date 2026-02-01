import streamlit as st
import pickle
import numpy as np
import os  # Add this

# Get the directory where app.py is located
base_path = os.path.dirname(__file__)

# Join the paths correctly
model_path = os.path.join(base_path, 'best_knn_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')

# Load the model and scaler using the full path
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        background-attachment: fixed;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Get an accurate estimate of your car\'s selling price based on key parameters</div>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📋 Car Details")

with st.sidebar.form("prediction_form"):
    st.subheader("Enter Car Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year of Purchase", min_value=2000, max_value=2024, value=2015, step=1)
        present_price = st.number_input("Present Price (₹ Lakhs)", min_value=0.0, value=5.0, step=0.1)
    
    with col2:
        kms_driven = st.number_input("Kilometers Driven", min_value=0, value=20000, step=1000)
        owner = st.selectbox("Previous Owners", [0, 1, 2, 3])
    
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    
    submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)

# Main content area
if submitted:
    # Data preprocessing
    fuel_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_petrol = 1 if fuel_type == 'Petrol' else 0
    seller_ind = 1 if seller_type == 'Individual' else 0
    trans_manual = 1 if transmission == 'Manual' else 0
    
    # Create feature array
    features = np.array([[year, present_price, kms_driven, owner, 
                          fuel_diesel, fuel_petrol, seller_ind, trans_manual]])
    
    # Scale numerical features
    features_scaled = features.copy()
    features_scaled[:, :4] = scaler.transform(features[:, :4])
    
    # Predict
    prediction = model.predict(features_scaled)
    output = round(prediction[0], 2)
    
    # Display result
    st.markdown(f'<div class="prediction-result">💰 Estimated Selling Price: ₹{output} Lakhs</div>', unsafe_allow_html=True)
    
    # Additional info
    st.info("💡 This prediction is based on historical data and market trends. Actual prices may vary.")

else:
    # Welcome message

    st.info("👈 Please fill in the car details in the sidebar and click 'Predict Price' to get an estimate.")
