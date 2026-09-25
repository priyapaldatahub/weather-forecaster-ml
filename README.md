# 🌦️ Weather Forecaster ML

A beginner-friendly Machine Learning project that predicts **average temperature** using historical weather data and provides predictions through an interactive **Streamlit web application**.

## 📌 Project Overview

Weather conditions can vary significantly based on location, season, precipitation, wind speed, and atmospheric pressure.

This project uses historical daily climate data to build a Machine Learning model that predicts **average temperature (°C)** based on selected weather and time-related features.

The complete workflow covers:

**Data Collection → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Model Training → Prediction → Streamlit Deployment**

---

## 🎯 Objectives

* Analyze historical weather data.
* Perform exploratory data analysis to identify weather patterns.
* Clean and preprocess weather data.
* Create useful date-based features.
* Encode categorical weather features.
* Train a Random Forest regression model.
* Save the trained model for prediction.
* Build an interactive Streamlit application.

---

## 🗂️ Dataset

**Dataset:** Global Daily Climate Data

**Source:** Kaggle — Global Daily Climate Data

The original dataset contains approximately **27.6 million daily weather records** across multiple locations.

### Original Features

* `station_id`
* `city_name`
* `date`
* `season`
* `avg_temp_c`
* `min_temp_c`
* `max_temp_c`
* `precipitation_mm`
* `snow_depth_mm`
* `avg_wind_dir_deg`
* `avg_wind_speed_kmh`
* `peak_wind_gust_kmh`
* `avg_sea_level_pres_hpa`
* `sunshine_total_min`

---

## 🔍 Exploratory Data Analysis

The EDA phase analyzed:

* Missing values
* Temperature distribution
* Seasonal temperature patterns
* City-wise temperature variations
* Precipitation patterns
* Wind speed distribution
* Weather feature relationships

### Key EDA Findings

* **Summer** had the highest average temperature among the seasons analyzed.
* **Winter** had the lowest average temperature.
* Faya-Largeau recorded the highest average temperature in the analyzed sample.
* Yakutsk recorded the lowest average temperature in the analyzed sample.
* Precipitation and wind-speed distributions contained extreme values.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline included:

1. Selected relevant weather features.
2. Handled missing values.
3. Converted the `date` column into datetime format.
4. Created date-based features:

   * Year
   * Month
   * Day
   * Day of Year
5. Selected categorical features:

   * City
   * Season
6. Applied One-Hot Encoding.
7. Split the dataset into training and testing sets.

### Dataset Split

| Dataset  | Records |
| -------- | ------: |
| Training | 309,763 |
| Testing  |  77,441 |
| Total    | 387,204 |

---

## 🤖 Machine Learning Model

### Random Forest Regressor

The project uses a **Random Forest Regression** model to predict average temperature.

### Input Features

* City
* Season
* Precipitation
* Average Wind Speed
* Average Sea Level Pressure
* Year
* Month
* Day
* Day of Year

### Target

`avg_temp_c`

---

## 📊 Model Pipeline

```text
Historical Weather Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Saved Model
        ↓
Streamlit Application
```

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can provide weather-related inputs and receive a predicted average temperature.

### Application Features

* City selection
* Season selection
* Weather feature inputs
* Temperature prediction
* Simple interactive interface

---

## 📁 Project Structure

```text
weather-forecaster-ml/
│
├── data/
│   ├── raw_data/
│   └── processed_data/
│
├── models/
│   ├── weather_model.pkl
│   ├── scaler.pkl
│   ├── city_categories.pkl
│   ├── season_categories.pkl
│   └── feature_columns.pkl
│
├── Notebook/
│
```
