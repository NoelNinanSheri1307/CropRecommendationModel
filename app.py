import streamlit as st
import joblib


model = joblib.load("crop_predictor.pkl")
scaler = joblib.load("scaler.pkl")
le_crop = joblib.load("crop_label_encoder.pkl")

def encode_soil(soil_str):
    mapping = {'Loamy': 1, 'Clay': 2, 'Sandy': 3,'Peaty':4,'Saline':5}
    return mapping.get(soil_str, 0)

def predict_crop(features):
    features_scaled = scaler.transform([features])
    crop_pred = model.predict(features_scaled)
    predicted_crop = le_crop.inverse_transform(crop_pred)
    return predicted_crop[0]


st.title("🌱 Crop Recommendation System")
st.markdown("""
    <style>
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>Developed by Noel Ninan Sheri</h5>", unsafe_allow_html=True)

st.header("📋 Enter Environmental Conditions")

# Soil Type (as numeric)
soil_type_label = st.selectbox("Soil Type", ['Loamy', 'Clay', 'Sandy','Peaty','Saline'])
soil_type = {'Loamy': 1, 'Clay': 2, 'Sandy': 3,'Peaty':4,'Saline':5}[soil_type_label]

# Other inputs
soil_ph = st.slider("Soil pH", min_value=4.5, max_value=9.0, value=6.5)
temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=20, value=10)

nitrogen = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=60)
phosphorus = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=40)
potassium = st.slider("Potassium (K)", min_value=0, max_value=100, value=50)

crop_yield = st.slider("Crop Yield (kg/ha)", min_value=0, max_value=100, value=30)

soil_quality = st.number_input("Soil Quality (float)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

features = [soil_type, soil_ph, temperature, humidity, wind_speed, 
            nitrogen, phosphorus, potassium, crop_yield, soil_quality]
if st.button("🌾 Predict Crop"):
    predicted_crop = predict_crop(features)
    st.success(f"System Recommends: Crop: **{predicted_crop}**")
