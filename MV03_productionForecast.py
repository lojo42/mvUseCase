import pandas as pd
from matplotlib import pyplot as plt
from darts import TimeSeries
from darts.models import Prophet


def load_package_data() -> pd.DataFrame:
    # Import data from csv files
    package_data = pd.read_csv("data/package_data.csv", sep=',', decimal='.')

    # Format data
    package_data.drop(["machine_identifier"], axis=1, inplace=True)
    package_data["timestamp"] = pd.to_datetime(package_data["timestamp"]).dt.tz_localize(None)
    package_data.sort_values(by="timestamp", inplace=True)
    package_data.set_index(package_data["timestamp"], inplace=True)

    return package_data


def forecast_production() -> None:
    # Using Darts with Facebook Prophet
    # https://github.com/unit8co/darts
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html#darts.models.forecasting.prophet_model.Prophet

    package_data = load_package_data()
    package_data = package_data.resample("8h", on="timestamp").agg("sum")  # Resample by 8-hour shift
    package_data = package_data[package_data.index.dayofweek < 5]   # Filter weekends
    package_data = package_data.drop("reject_packs", axis=1)  # Not relevant for the prediction
    package_data.reset_index(inplace=True)

    # Create Darts Timeseries Object (basically a Pandas DF in a wrapper)
    series = TimeSeries.from_dataframe(package_data, "timestamp", "good_packs", freq="8h")

    # Create the Prophet model
    # seasonal_periods = 15 -> 5 workdays * 3 shifts
    # fourier_order determines how fast the seasonality changes. Higher values lead to overfitting
    model = Prophet(
        add_seasonalities={
            "name": "workweek",
            "seasonal_periods": 15,
            "fourier_order": 3
        },
    )
    model.fit(series)

    # Predicting 120 shifts into the future -> 40 working days
    pred = model.predict(120)

    series.plot()
    pred.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.show()

    prediction_df = pred.pd_dataframe()
    del series

    correct_index = pd.bdate_range("2022-08-01", periods=8, freq="W-Mon")
    prediction_df = prediction_df.resample("5D").agg("mean")
    prediction_df.index = correct_index
    prediction_df.reset_index(inplace=True)
    prediction_df.rename(columns={
        "index": "Woche_von",
        "good_packs": "Gutpackungen_pro_Schicht",
    }, inplace=True)
    prediction_df.to_csv("output/ICC_Produktionsvorhersage.csv", sep=";", decimal=",", index=False, float_format='%.0f')
    del prediction_df
    # Over the next 8 weeks, we forecast the customer to produce an average of roughly 30,000 packs per shift


forecast_production()
