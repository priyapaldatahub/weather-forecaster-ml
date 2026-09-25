import pandas as pd


def load_weather_data(file_path):
    """
    Load the weather dataset from a Parquet file.
    """

    df = pd.read_parquet(file_path)

    print("Weather dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    return df


if __name__ == "__main__":

    file_path = "../data/raw_data/daily_weather.parquet"

    df = load_weather_data(file_path)

    print("\nFirst 5 rows:")
    print(df.head())