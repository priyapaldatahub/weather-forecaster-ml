import pandas as pd


def load_data(file_path):
    """Load the weather dataset."""

    df = pd.read_parquet(
        file_path,
        columns=[
            "city_name",
            "date",
            "season",
            "precipitation_mm",
            "avg_wind_speed_kmh",
            "avg_sea_level_pres_hpa",
            "avg_temp_c"
        ]
    )

    return df


def clean_data(df):
    """Clean missing values from the dataset."""

    # Remove rows where city or target temperature is missing
    df = df.dropna(
        subset=["city_name", "avg_temp_c"]
    ).copy()

    # Numerical columns
    numeric_columns = [
        "precipitation_mm",
        "avg_wind_speed_kmh",
        "avg_sea_level_pres_hpa"
    ]

    # Fill missing numerical values with median
    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


def create_features(df):
    """Create date and categorical features."""

    # Date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    # Convert categorical columns to category codes
    df["city_code"] = (
        df["city_name"]
        .astype("category")
        .cat.codes
    )

    df["season_code"] = (
        df["season"]
        .astype("category")
        .cat.codes
    )

    # Remove columns no longer needed
    df = df.drop(
        columns=["date", "city_name", "season"]
    )

    return df


def prepare_data(file_path, sample_size=200000):
    """Complete preprocessing pipeline."""

    df = load_data(file_path)

    # Take sample for faster processing
    df = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    df = clean_data(df)

    df = create_features(df)

    return df