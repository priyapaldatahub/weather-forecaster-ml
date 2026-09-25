import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_data, clean_data, create_features


# ==========================================
# 1. Project Paths
# ==========================================

DATA_PATH = "../data/raw_data/daily_weather.parquet"
MODEL_DIR = "../models"


# ==========================================
# 2. Load Dataset
# ==========================================

print("Loading dataset...")

df = load_data(DATA_PATH)

print("Dataset loaded successfully!")
print("Original Shape:", df.shape)


# ==========================================
# 3. Create Same Sample Used for Training
# ==========================================

print("\nCreating sample...")

df = df.sample(
    n=200000,
    random_state=42
).copy()

print("Sample Shape:", df.shape)


# ==========================================
# 4. Clean Data
# ==========================================

print("\nCleaning data...")

df = clean_data(df)

print("Cleaning completed!")
print("Shape after cleaning:", df.shape)


# ==========================================
# 5. Save Category Mappings
# ==========================================

city_categories = (
    df["city_name"]
    .astype("category")
    .cat.categories
    .tolist()
)

season_categories = (
    df["season"]
    .astype("category")
    .cat.categories
    .tolist()
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    city_categories,
    os.path.join(MODEL_DIR, "city_categories.pkl")
)

joblib.dump(
    season_categories,
    os.path.join(MODEL_DIR, "season_categories.pkl")
)

print("City and season mappings saved!")


# ==========================================
# 6. Feature Engineering
# ==========================================

print("\nCreating features...")

df = create_features(df)

print("Feature creation completed!")


# ==========================================
# 7. Separate Features and Target
# ==========================================

X = df.drop(
    columns=["avg_temp_c"]
)

y = df["avg_temp_c"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 8. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ==========================================
# 9. Create Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=30,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nRandom Forest model created!")


# ==========================================
# 10. Train Model
# ==========================================

print("\nModel training started...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 11. Make Predictions
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("Predictions completed!")


# ==========================================
# 12. Evaluate Model
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))


# ==========================================
# 13. Save Model
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "weather_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")
print(model_path)


# ==========================================
# 14. Save Feature Columns
# ==========================================

feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)

print("Feature columns saved successfully!")


# ==========================================
# 15. Final Message
# ==========================================

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("===================================")