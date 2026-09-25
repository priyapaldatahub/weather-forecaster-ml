# 🌤️ Weather Forecaster ML

A beginner-friendly Machine Learning project that predicts average temperature using historical global weather data.

## 📌 Project Overview

This project uses historical daily weather data to build a Machine Learning regression model for predicting average temperature.

The complete workflow includes:

- Data collection
- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering
- Categorical encoding
- Train-test splitting
- Random Forest regression
- Model evaluation
- Model saving using Joblib
- Streamlit web application

## 🎯 Objective

To build a Machine Learning model that predicts `avg_temp_c` using historical weather information such as:

- City
- Season
- Precipitation
- Average wind speed
- Sea level pressure
- Date-based features

## 📊 Dataset

Dataset: Global Daily Climate Data

Source: Kaggle

Dataset contains daily weather observations from cities around the world.

Original dataset size:

- 27.6 million+ records
- 14 features
- 1,234 cities

## 🛠️ Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Random Forest Regressor

### Model Deployment
- Streamlit

### Model Serialization
- Joblib

### Development Tools
- Jupyter Notebook
- VS Code

## 🔄 Project Workflow

```text
Raw Weather Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Save Model
        ↓
Streamlit Application