import pandas as pd
from datetime import timedelta


def load_package_data() -> pd.DataFrame:
    # Import data from csv files
    package_data = pd.read_csv("data/package_data.csv", sep=',', decimal='.')

    # Format data
    package_data.drop(["machine_identifier"], axis=1, inplace=True)
    package_data["timestamp"] = pd.to_datetime(package_data["timestamp"])
    package_data.sort_values(by="timestamp", inplace=True)
    package_data.set_index(package_data["timestamp"], inplace=True)

    return package_data


def load_error_data() -> pd.DataFrame:
    # Import data from csv files
    error_data = pd.read_csv("data/error_messages_timeline.csv", sep=',', decimal='.')

    # Format data
    error_data.drop(["machine_identifier"], axis=1, inplace=True)
    error_data.sort_values(by="start_ts", inplace=True)
    error_data["start_ts"] = pd.to_datetime(error_data["start_ts"])
    error_data["end_ts"] = pd.to_datetime(error_data["end_ts"])

    return error_data


def load_recipe_data() -> pd.DataFrame:
    # Import data from csv files
    recipe_data = pd.read_csv("data/recipe_data.csv", sep=',', decimal='.')

    # Format data
    recipe_data.drop(["machine_identifier"], axis=1, inplace=True)
    recipe_data.sort_values(by="timestamp", inplace=True)
    recipe_data["timestamp"] = pd.to_datetime(recipe_data["timestamp"])
    recipe_data.set_index(recipe_data["timestamp"], inplace=True)

    return recipe_data


def reject_pack_analysis() -> None:
    package_data = load_package_data()

    # Our sales rep wants to know "when" we produced the most reject packs. "When" is a rather unprecise request, so
    # we'll provide him with three different dimensions:
    # 1. 24-hour representation so he can analyse different shifts
    # 2. weekday representation to see if Mondays for example are an issue
    # 3. weekly representation to see if certain calendar weeks perform worse than others
    # Note: we could aggregate by either sum or mean (or both). They would provide us with different information.
    # mean provides an overview of the good vs reject pack rate. sum provides an overall view of production capacity.
    # Since the sales rep asked for the amount of reject packs specifically, we'll just go with sum. We could always
    # provide a mean aggregation later to analyse if the rate of good vs reject packs is affected by different shifts
    # or workdays etc.

    # Hourly
    package_data.loc[:, "hour"] = package_data["timestamp"].dt.hour
    package_data_hourly = package_data.groupby(["hour"])[["good_packs", "reject_packs"]].sum()
    package_data_hourly.index.names = ["Stunde"]
    package_data_hourly.rename(columns={
        "good_packs": "Gutpackungen",
        "reject_packs": "Schlechtpackungen",
    }, inplace=True)
    package_data_hourly["Rate_Schlechtpackungen"] = (
            package_data_hourly["Schlechtpackungen"] / package_data_hourly["Gutpackungen"])
    package_data_hourly.to_csv("output/ICC_Schlechtpackungen_Stuendlich.csv", sep=";", decimal=",", index=True,
                               float_format='%.3f')
    del package_data_hourly

    # Weekday
    package_data.loc[:, "weekday"] = package_data["timestamp"].dt.weekday
    package_data_weekday = package_data.groupby(["weekday"])[["good_packs", "reject_packs"]].sum()
    package_data_weekday.set_index([["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]],
                                   inplace=True)
    package_data_weekday.index.names = ["Wochentag"]
    package_data_weekday.rename(columns={
        "good_packs": "Gutpackungen",
        "reject_packs": "Schlechtpackungen",
    }, inplace=True)
    package_data_weekday["Rate_Schlechtpackungen"] = (
            package_data_weekday["Schlechtpackungen"] / package_data_weekday["Gutpackungen"])
    package_data_weekday.to_csv("output/ICC_Schlechtpackungen_Wochentag.csv", sep=";", decimal=",", index=True,
                                float_format='%.3f')
    del package_data_weekday

    # Weekly
    package_data_weekly = package_data.resample("1W")[["good_packs", "reject_packs"]].agg("sum")
    package_data_weekly.index = package_data_weekly.index-timedelta(6)
    package_data_weekly.index = package_data_weekly.index.strftime('%d.%m.%Y')
    package_data_weekly.index.names = ["Woche_von"]
    package_data_weekly.rename(columns={
        "good_packs": "Gutpackungen",
        "reject_packs": "Schlechtpackungen",
    }, inplace=True)
    package_data_weekly["Rate_Schlechtpackungen"] = (
            package_data_weekly["Schlechtpackungen"] / package_data_weekly["Gutpackungen"])
    package_data_weekly.to_csv("output/ICC_Schlechtpackungen_Woechentlich.csv", sep=";", decimal=",", index=True,
                               float_format='%.3f')
    del package_data_weekly

    # Our sales rep also mentioned that a certain recipe might produce more issues than others. So we load the recipe
    # data, merge it with the package data and then aggregate the rate of reject packs for each different recipe.
    # We see that they do perform similar. And most importantly "vegan parmesan" performs better than the others.
    recipe_data = load_recipe_data()
    package_data = pd.merge_asof(left=package_data, right=recipe_data, left_index = True, right_index = True,
                                 direction="backward")
    del recipe_data

    package_data = package_data.drop(["timestamp_y"], axis=1)
    package_data_by_recipe = package_data.groupby(["recipe"])[["good_packs", "reject_packs"]].sum()
    package_data_by_recipe["Rate_Schlechtpackungen"] = (
            package_data_by_recipe["reject_packs"] / package_data_by_recipe["good_packs"])
    package_data_by_recipe.index.names = ["Rezept"]
    package_data_by_recipe.rename(columns={
        "good_packs": "Gutpackungen",
        "reject_packs": "Schlechtpackungen",
    }, inplace=True)
    package_data_by_recipe.to_csv("output/ICC_Schlechtpackungen_NachRezept.csv", sep=";", decimal=",", index=True,
                                  float_format='%.3f')
    del package_data_by_recipe

    # Now we just need to look at the error hypothesis of the sales rep.
    # In order to test the rep's hypothesis, we create two aggregations:
    # 1. The total count of any given error code - i.e. which error occurs most often
    # 2. The total error duration per error code - i.e. which error causes the most downtime in total
    # We notice that Error 1019 is infact the most common, but others, like error 2030 result in way more
    # overall downtime of the machine.
    error_data = load_error_data()
    error_data = error_data.groupby(error_data["code"])[["code", "duration_in_s"]].agg({
      "code": "count",
      "duration_in_s": "sum"
    })
    error_data.index.names = ["Error_Code"]
    error_data.rename(columns={
        "code": "Anzahl_Meldungen",
        "duration_in_s": "Gesamtdauer_Störung_Sekunden",
    }, inplace=True)
    error_data.to_csv("output/ICC_Fehlercodeanalyse.csv", sep=";", decimal=",", index=True, float_format='%.2f')
    del error_data
    del package_data


reject_pack_analysis()
