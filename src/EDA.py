import pandas as pd
import matplotlib.pyplot as plt


def perform_eda(file_path):
    """
    Perform basic exploratory data analysis
    on the weather dataset.
    """

    df = pd.read_parquet(
        file_path,
        columns=[
            "city_name",
            "date",
            "season",
            "avg_temp_c",
            "precipitation_mm",
            "avg_wind_speed_kmh"
        ]
    )

    print("Dataset Shape:", df.shape)

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())

    # Sample for faster visualization
    sample = df.sample(
        n=min(200000, len(df)),
        random_state=42
    )

    # Seasonal temperature
    seasonal_temperature = (
        sample.groupby("season")["avg_temp_c"]
        .mean()
        .sort_values()
    )

    print("\nAverage Temperature by Season:")
    print(seasonal_temperature)

    # Monthly temperature
    sample["month"] = sample["date"].dt.month

    monthly_temperature = (
        sample.groupby("month")["avg_temp_c"]
        .mean()
    )

    print("\nAverage Temperature by Month:")
    print(monthly_temperature)

    # Temperature plot
    monthly_temperature.plot(
        kind="line",
        marker="o"
    )

    plt.title("Average Monthly Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    file_path = "../data/raw_data/daily_weather.parquet"

    perform_eda(file_path)