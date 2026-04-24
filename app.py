import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_house_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Set up the page layout
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

# Main Header
st.title("🏡 California House Price Predictor")
st.markdown("""
This app predicts the **median house value** in California districts based on various features.
It uses an **Ensemble Learning technique (Random Forest)** trained on the California Housing dataset.
""")

st.divider()

if model is None:
    st.error("Model file `rf_house_model.pkl` not found. Please train and save the model first.")
else:
    # Input Features
    st.header("Enter Housing Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.slider("Median Income (in tens of thousands)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
        house_age = st.slider("House Age (years)", min_value=1, max_value=100, value=25, step=1)
        ave_rooms = st.slider("Average Rooms", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        ave_bedrms = st.slider("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
    with col2:
        population = st.slider("Population", min_value=10, max_value=5000, value=1000, step=10)
        ave_occup = st.slider("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        latitude = st.slider("Latitude", min_value=32.0, max_value=42.0, value=35.0, step=0.1)
        longitude = st.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-119.0, step=0.1)

    # Predict Button
    if st.button("Predict Price", type="primary"):
        # Create input array
        input_data = pd.DataFrame({
            'MedInc': [med_inc],
            'HouseAge': [house_age],
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f"### Predicted Median House Value: ${prediction * 100000:,.2f}")
        st.balloons()

st.divider()
st.caption("Ensemble Learning Techniques Task Implementation")
