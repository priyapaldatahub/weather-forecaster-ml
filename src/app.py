import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Weather Forecaster",
    page_icon="🌤️",
    layout="centered"
)


# ==========================================
# Model Paths
# ==========================================
# ==========================================
# Model Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "weather_model.pkl"
)

CITY_PATH = os.path.join(
    BASE_DIR,
    "models",
    "city_categories.pkl"
)

SEASON_PATH = os.path.join(
    BASE_DIR,
    "models",
    "season_categories.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


# ==========================================
# Load Model and Mappings
# ==========================================

def load_model():

    model = joblib.load(MODEL_PATH)
    cities = joblib.load(CITY_PATH)
    seasons = joblib.load(SEASON_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, cities, seasons, feature_columns


model, cities, seasons, feature_columns = load_model()


# ==========================================
# Application Title
# ==========================================

st.title("🌤️ Weather Forecaster")

st.write(
    "Predict average temperature using historical weather information."
)


# ==========================================
# User Inputs
# ==========================================

st.subheader("Enter Weather Information")


city = st.selectbox(
    "Select City",
    cities
)


season = st.selectbox(
    "Select Season",
    seasons
)


precipitation = st.number_input(
    "Precipitation (mm)",
    min_value=0.0,
    value=0.0
)


wind_speed = st.number_input(
    "Average Wind Speed (km/h)",
    min_value=0.0,
    value=10.0
)


pressure = st.number_input(
    "Average Sea Level Pressure (hPa)",
    min_value=800.0,
    max_value=1100.0,
    value=1013.0
)


year = st.number_input(
    "Year",
    min_value=1750,
    max_value=2100,
    value=2026,
    step=1
)


month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)


day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1,
    step=1
)


day_of_year = st.number_input(
    "Day of Year",
    min_value=1,
    max_value=366,
    value=1,
    step=1
)


# ==========================================
# Convert Categories to Codes
# ==========================================

city_code = cities.index(city)
season_code = seasons.index(season)


# ==========================================
# Create Input Data
# ==========================================

input_data = pd.DataFrame({
    "precipitation_mm": [precipitation],
    "avg_wind_speed_kmh": [wind_speed],
    "avg_sea_level_pres_hpa": [pressure],
    "year": [year],
    "month": [month],
    "day": [day],
    "day_of_year": [day_of_year],
    "city_code": [city_code],
    "season_code": [season_code]
})


# Make sure feature order matches training
input_data = input_data[
    feature_columns
]


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Temperature"):

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"Predicted Average Temperature: {prediction:.2f} °C"
    )