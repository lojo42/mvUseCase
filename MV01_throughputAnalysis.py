import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta


def load_oee_data() -> pd.DataFrame:
    # Import data from csv files
    oee_data = pd.read_csv("data/oee_data.csv", sep=',', decimal='.')

    # Format data
    oee_data.drop(["machine_identifier"], axis=1, inplace=True)
    oee_data["timestamp"] = pd.to_datetime(oee_data["timestamp"])
    oee_data.sort_values(by="timestamp", inplace=True)

    # Fill column so we can later on determine the deviation between programmed and actual throughput
    oee_data["expected_cycles_per_minute"] = oee_data["expected_cycles_per_minute"].ffill()

    # Drop first few rows, as we don't know the programmed throughput at the beginning of the logfile.
    # As we will do averages later on, dropping those lines should not lead to false results.
    oee_data.dropna(inplace=True)

    # Create new deviation metric
    oee_data["deviation_from_programmed_throughput"] = \
        (oee_data["expected_cycles_per_minute"] - oee_data["actual_cycles_per_minute"]).abs()

    return oee_data


def load_package_data() -> pd.DataFrame:
    # Import data from csv files
    package_data = pd.read_csv("data/package_data.csv", sep=',', decimal='.')

    # Format data
    package_data.drop(["machine_identifier"], axis=1, inplace=True)
    package_data["timestamp"] = pd.to_datetime(package_data["timestamp"])
    package_data.sort_values(by="timestamp", inplace=True)

    return package_data


def throughput_analysis() -> None:
    # We begin by loading the data into a DataFrame and doing some basic formatting
    # We then do a first visual inspection of the data and notice that the timestamps are not
    # perfectly distributed.
    oee_data = load_oee_data()

    # In order to determine if we can still use the metric as an indicator for the average
    # throughput, we do a quick count-aggregation into 5 minute windows and plot the result.
    # The plot shows a somewhat reasonable distribution of timestamps across the entire logfile,
    # so doing average-aggregations of the "actual_cycles_per_minute" values should yield usable results.
    oee_data_resampled = oee_data.resample("5min", on="timestamp").agg("count")
    plt.plot(oee_data_resampled.index, oee_data_resampled["expected_cycles_per_minute"], "bo")
    plt.show()
    del oee_data_resampled  # We don't need that anymore, so no use keeping it in memory.

    # In order to provide quickly comprehensible results, we do weekly aggregations.
    # Looking at a 13-week time period, this should be a good balance between granularity and readability.
    oee_data_weekly = oee_data.resample("1W", on="timestamp").agg("mean")
    del oee_data

    # Our sales rep also wants to know the amount of produced packs. This should be a pretty straight forward
    # aggregation. We load the data, do some quick formatting and then aggregate on a weekly basis as well.
    package_data = load_package_data()
    package_data_weekly = package_data.resample("1W", on="timestamp").agg("sum")
    del package_data

    # Now we just need to merge the aggregated package data into the oee data.
    oee_data_weekly = oee_data_weekly.join(package_data_weekly)
    del package_data_weekly

    # It's not worth doing a dashboard or report for a one time analysis, so we plan on providing our sales rep with an
    # Excel file. Prior, we drop some data, insert the planned Cycles as new column and do some basic formatting, so
    # it's quicker for him to understand.
    oee_data_weekly.index = oee_data_weekly.index-timedelta(6)
    oee_data_weekly.index = oee_data_weekly.index.strftime('%d.%m.%Y')
    oee_data_weekly.index.names = ["Woche_von"]
    oee_data_weekly.drop("deviation_from_programmed_throughput", axis=1, inplace=True)
    oee_data_weekly.insert(0, "Geplante_Taktleistung", 10)
    oee_data_weekly.rename(columns={
        "expected_cycles_per_minute": "Programmierte_Taktleistung",
        "actual_cycles_per_minute": "Tatsächliche_Taktleistung",
        "good_packs": "Gutpackungen",
        "reject_packs": "Schlechtpackungen"
    }, inplace=True)
    oee_data_weekly.to_csv("output/ICC_InitialeAnalyse.csv", sep=";", decimal=",", index=True,
                           float_format='%.2f')
    del oee_data_weekly


throughput_analysis()
