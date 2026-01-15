from src.aux_utilities import extracting_year, extracting_month, extracting_day

import pandas as pd
import os
import json
from dotenv import load_dotenv
import numpy as np
from src.aux_utilities import add_weather_data_to_combined
from src.aux_utilities import remove_weekends_from_comined_datasets, remove_NaNs_and_zeros_from_datasets
import holidays
from tqdm import tqdm
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")

print("Creating total strava enhanced dataset...")
"""
We create the total strava enhanced dataset by normalising the strava data per counter region
and adding weather and temporal features
"""
strava_total = pd.read_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total_locationid.csv")
strava_total['date'] = pd.to_datetime(strava_total['date'])
strava_total['counter_region_index'] = strava_total['counter_region_index'].astype(int)

strava_total = extracting_year(strava_total)
strava_total = extracting_month(strava_total)
strava_total = extracting_day(strava_total)
strava_total = strava_total[['edge_uid','date', 'year', 'month', 'day', 'counter_region_index', 'strava_trips']]

"""
Getting unique counter region indices
"""
counter_indices = (strava_total['counter_region_index']).unique().tolist()

"""
Loading normalisation values
"""
with open(f"{DATA_PATH}/bike_counters/_max_min_matched_strava_edges.json", "r") as f:
    max_min_matched_strava_edges = json.load(f)

"""
Weather Data
"""
weather_data = pd.read_csv(f"{DATA_PATH}/weather/weather_data_complete.csv")
weather_data['date'] = pd.to_datetime(weather_data['date'])


strava_total['weekday'] = strava_total['date'].dt.weekday + 1  # Monday=1, Sunday=7

strava_total= add_weather_data_to_combined(strava_total,weather_data=weather_data)
    
strava_total = remove_weekends_from_comined_datasets(strava_total)
strava_total = remove_NaNs_and_zeros_from_datasets(strava_total,False)


strava_total['strava_trips_normalised'] = None

print("== Normalising strava trips per counter region ==")
for bike_counter_id in counter_indices:
    bike_counter_id = int(bike_counter_id)
    print(f"   Normalising for bike counter region index: {bike_counter_id}")
    max_value = max_min_matched_strava_edges[str(bike_counter_id + 1)]['max']
    min_value = max_min_matched_strava_edges[str(bike_counter_id + 1)]['min']
    
    strava_total.loc[strava_total['counter_region_index'] == (bike_counter_id),'strava_trips_normalised'] = (strava_total[strava_total['counter_region_index'] == bike_counter_id]['strava_trips'] - min_value )/(max_value - min_value)

print("Adding normalised weather and temporal features...")
for feature in ['temp_avg','precip_quantity','wind_speed_10m','year']:
    strava_total[feature+'_normalised'] = (strava_total[feature] - strava_total[feature].min())/(strava_total[feature].max()-strava_total[feature].min())

print("Adding cyclical temporal features...")
print(" - weekday cyclical features...")
strava_total["wd_sin"] = np.sin(2 * np.pi * (strava_total['weekday']-1) / 5)
strava_total["wd_cos"] = np.cos(2 * np.pi * (strava_total['weekday']-1) / 5)

print(" - month cyclical features...")
strava_total['month_sin'] = np.sin(2 * np.pi * (strava_total['month']-1) / 12)
strava_total['month_cos'] = np.cos(2 * np.pi * (strava_total['month']-1) / 12)

strava_total["any_rain"] = (strava_total["precip_quantity"] > 0).astype(int)


years = strava_total['year'].unique().tolist()
be_holidays = holidays.BE(years=years)

strava_total['is_holiday'] = strava_total['date'].dt.date.apply(lambda d: 1 if d in be_holidays else 0)
print("Saving total strava enhanced dataset...")
strava_total.to_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total_enhanced.csv",index = False)

print("Total strava enhanced dataset created and saved.")

print("Saving yearly splits of the total strava enhanced dataset...")
for year in [2019,2020,2021,2022,2023,2024]:
    strava_total.loc[strava_total['year'] == year].to_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total_enhanced_{year}.csv",index = False)

print("Yearly splits saved.")