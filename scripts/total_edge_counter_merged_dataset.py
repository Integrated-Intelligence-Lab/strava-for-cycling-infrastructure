import pandas as pd
import math as math
from tqdm import tqdm

import numpy as np

import pickle

import os
import json
import holidays
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")

from src.loading_datasets import loading_brussels_shape, loading_bike_counters
from src.aux_utilities import extracting_year, extracting_month


from src.aux_utilities import combine_counter_and_strava, add_weather_data_to_combined
from src.aux_utilities import remove_weekends_from_comined_datasets, remove_NaNs_and_zeros_from_datasets

if __name__ == "__main__":
    print("Loading datasets...")
    # 1 Weather Data
    weather_data = pd.read_csv(f"{DATA_PATH}/weather/weather_data_complete.csv")
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    # 2a Bike counter ids
    with open(f'{DATA_PATH}/bike_counters/_bike_counters_IDs.json') as json_file:
        bike_counter_ids = json.load(json_file)
    # 2b Bike counters with locations and nearby edges
    bike_counters_gdf = loading_bike_counters()

    # 3 Brussels Shape
    brussels_region_shape, _ = loading_brussels_shape()

    # 4 Strava Edge Data with location ids
    strava_edge_total_locationid = pd.read_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total_locationid.csv")

    merged_datasets = {}
    max_min_matched_strava_edges = {}
    max_min_per_counter = {}
    """
    Merging the datasets per bike counter
    """
    for idx,device_name in tqdm(enumerate(bike_counter_ids),desc = "Merging datasets per bike counter"):
        device_idx  = int(idx + 1)
        max_min_per_counter[device_idx] = {}
        max_min_matched_strava_edges[device_idx] = {}
        # 1 Merge Strava And Counter
        counter_strava_combined = combine_counter_and_strava(device_name,strava_dataset=strava_edge_total_locationid,gdf_bike_counters=bike_counters_gdf)

        # 2 Add Weather Data
        edges_device_weather_combined = add_weather_data_to_combined(counter_strava_combined,weather_data=weather_data)

        # 3 Clean Data
        ## a) Remove weekends
        combined_temp = remove_weekends_from_comined_datasets(edges_device_weather_combined)

        ## b) Remove NaNs and zeros
        combined_temp = remove_NaNs_and_zeros_from_datasets(combined_temp)

        ## c) Extract year and month into their own columns
        combined_temp = extracting_year(combined_temp)
        combined_temp = extracting_month(combined_temp)

        complete_merged_normalised = combined_temp.copy(deep = True)
        
    
        # 4 Normalisation and transformations: For Strava and Counter trips we normalise per counter. However further down we will normalise weazther data globally over all counters
        
        max_train_fraction_strava = complete_merged_normalised['strava_trips'].max()
        min_train_fraction_strava = complete_merged_normalised['strava_trips'].min()
        max_min_matched_strava_edges[device_idx]["max"] = max_train_fraction_strava
        max_min_matched_strava_edges[device_idx]["min"] = min_train_fraction_strava

        max_train_fraction_counter = complete_merged_normalised['counter_trips'].max()
        min_train_fraction_counter = complete_merged_normalised['counter_trips'].min()
        max_min_per_counter[device_idx]["max"] = max_train_fraction_counter
        max_min_per_counter[device_idx]["min"] = min_train_fraction_counter
        

        ## a) Min-Max Normalisation
        complete_merged_normalised['strava_trips_normalised'] = (complete_merged_normalised['strava_trips'] - min_train_fraction_strava)/(max_train_fraction_strava - min_train_fraction_strava)
        complete_merged_normalised['counter_trips_normalised'] = (complete_merged_normalised['counter_trips'] - min_train_fraction_counter)/(max_train_fraction_counter - min_train_fraction_counter)
        

        ## b) Log-transform 
        complete_merged_normalised['counter_trips_log'] = np.log1p(complete_merged_normalised['counter_trips'])
        complete_merged_normalised['strava_trips_log'] = np.log1p(complete_merged_normalised['strava_trips'])

        ## c) Online standardisation using an EWMA for causal, rolling mean and std
        alpha = 0.1  # EWMA decay factor 
        for col in ['counter_trips', 'strava_trips']:
            ewma_mean = complete_merged_normalised[col].ewm(alpha=alpha, adjust=False).mean()
            ewma_std = complete_merged_normalised[col].ewm(alpha=alpha, adjust=False).std().replace(0, np.nan)
            complete_merged_normalised[f'{col}_ewma_norm'] = (complete_merged_normalised[col] - ewma_mean) / ewma_std

        merged_datasets[device_name] = complete_merged_normalised
    
    with open(f'{DATA_PATH}/merged_datasets_per_counter.pkl', 'wb') as f:
        pickle.dump(merged_datasets, f)

    print("Merged datasets saved successfully.")

    """
    Merged dataset over all counters: This one will be used for training further on
    1. Normalisation of weather data & Year over all counters
    2. Encoding of month & week as cyclical feature
    3. Adding holiday information
    4. Add a binary rain flag
    5. Saving the final merged dataset
    """
    total_merged_dataset = pd.concat([*merged_datasets.values()]) 

    # 1. Normalisation of weather data & Year over all counters
    for feature in ['temp_avg','precip_quantity','wind_speed_10m','year']:
        total_merged_dataset[feature+'_normalised'] = (total_merged_dataset[feature] - total_merged_dataset[feature].min())/(total_merged_dataset[feature].max()- total_merged_dataset[feature].min())

    # 2. Encoding of month & week as cyclical feature:Weekday: 1-5 (Mon-Fri) and Months: 1-12
    total_merged_dataset["wd_sin"] = np.sin(2 * np.pi * (total_merged_dataset['weekday']-1) / 5)
    total_merged_dataset["wd_cos"] = np.cos(2 * np.pi * (total_merged_dataset['weekday']-1) / 5)

    total_merged_dataset['month_sin'] = np.sin(2 * np.pi * (total_merged_dataset['month']-1) / 12)
    total_merged_dataset['month_cos'] = np.cos(2 * np.pi * (total_merged_dataset['month']-1) / 12)

    
    years = total_merged_dataset['year'].unique().tolist()
    be_holidays = holidays.BE(years=years)

    # 3. Create the is_holiday flag (1 if holiday, 0 otherwise)
    total_merged_dataset['is_holiday'] = total_merged_dataset['date'].dt.date.apply(lambda d: 1 if d in be_holidays else 0)

    # 4. Create a binary rain flag (1 if precipitation > 0, 0 otherwise)

    total_merged_dataset["any_rain"] = (total_merged_dataset["precip_quantity"] > 0).astype(int)

    # 5. Saving the final merged dataset
    total_merged_dataset.to_csv(f"{DATA_PATH}/total_merged_dataset.csv",index = False)
    """ 
    Saving the max and min per counter 
    We will use this for counter and strava trips for inverse normalisation later on 
    """
    

    with open(f"{DATA_PATH}/bike_counters/_max_min_per_counter.json", "w") as f:
        json.dump(max_min_per_counter, f)

    with open(f"{DATA_PATH}/bike_counters/_max_min_matched_strava_edges.json", "w") as f:
        json.dump(max_min_matched_strava_edges, f)