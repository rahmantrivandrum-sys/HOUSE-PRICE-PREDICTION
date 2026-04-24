# HOUSE-PRICE-PREDICTION


🏡 California House Price Prediction App

This project is a Machine Learning web application built using Streamlit that predicts median house prices in California districts using an Ensemble Learning technique (Random Forest).

🚀 Features
Predicts house prices based on user inputs
Uses Random Forest (Ensemble Learning) model
Interactive UI built with Streamlit
Real-time predictions with sliders
Clean and simple interface
📊 Input Features

The model takes the following inputs:

Median Income (MedInc)
House Age (HouseAge)
Average Rooms (AveRooms)
Average Bedrooms (AveBedrms)
Population
Average Occupancy (AveOccup)
Latitude
Longitude
🧠 Model Used
Algorithm: Random Forest Regressor
Technique: Ensemble Learning
Trained on the California Housing Dataset
🖥️ How to Run the Project
1. Install Requirements
pip install streamlit pandas numpy scikit-learn joblib
2. Ensure Model File Exists

Make sure the trained model file is present:

rf_house_model.pkl
3. Run the App
streamlit run app.py
📁 Project Structure
├── app.py                      # Streamlit app
├── rf_house_model.pkl          # Trained model
├── Ensemble_Learning_Techniques.ipynb  # Model training notebook
├── README.md                  # Project documentation
