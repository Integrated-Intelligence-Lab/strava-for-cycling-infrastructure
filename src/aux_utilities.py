import numpy as np
import pandas as pd

import os
import json
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")

def w_clipped_gauss(d, sigma=7, tau=2.0):
    """Totally flat up to sigma, Gaussian beyond."""
    d = np.asarray(d)
    out = np.ones_like(d, dtype=float)
    mask = d > sigma
    out[mask] = np.exp(-0.5 * ((d[mask] - sigma) / tau) ** 2)
    return out

def w_logistic(d, sigma=7, k=1.0):
    """Flatter until sigma, then sigmoid drop; k = metres that move weight from 0.73 to 0.27."""
    return 1.0 / (1.0 + np.exp((d - sigma) / k))


def extracting_year(total_data):
    total_data['year'] = total_data['date'].dt.year
    return total_data
def extracting_month(total_data):
    total_data['month'] = total_data['date'].dt.month
    return total_data
def extracting_day(total_data):
    total_data['day'] = total_data['date'].dt.day
    return total_data

def extracting_nearby_edgesuid(device_name,gdf_bike_counters):
    counter_row = gdf_bike_counters.loc[gdf_bike_counters['device_name'] == device_name]

    # We extract the nearby edges from  the dataframe and remove possible NaN values since not all devices have relevant parallel edges
    counter_nearby_edgesuid = counter_row[['edgeUID checked','Parallel edgeUID','2nd Parallel edgeUID']].copy()
    counter_nearby_edgesuid.dropna(axis=1,inplace = True)
    return counter_nearby_edgesuid.values[0]

def loading_and_grouping_counter_history(device_name,gdf_bike_counters):

    # We load the history_csv file of the corresponding counter
    counter_history =  pd.read_csv(f"{DATA_PATH}/bike_counters/raw/{device_name}_history.csv")
    # history.csv files contain quarterly information. So we sum these together per day
    counter_history_daily = counter_history.groupby('Date').sum()['Count'].reset_index()
    counter_history_daily.rename(columns = {'Date': 'date','Count':'counter_trips'},inplace = True)
    # Transform the date column to datetime in order to compare it with the strava data
    counter_history_daily['date'] = pd.to_datetime(counter_history_daily['date'])   
    counter_history_daily['identifier_counter'] = device_name
    counter_history_daily['identifier_counter_int'] = int(gdf_bike_counters.loc[gdf_bike_counters['device_name'] == device_name]['id'].unique()[0])
    return counter_history_daily

def aggregating_nearby_strava_edges(device_name,strava_dataset,gdf_bike_counters):
    # We extract the nearby edges from the dataframe
    counter_nearby_edgesuid = extracting_nearby_edgesuid(device_name,gdf_bike_counters)

    # We select the relevant edges from the strava data by checking if the edge_uid is in the list of nearby edges
    device_edges = strava_dataset.loc[strava_dataset["edge_uid"].isin(counter_nearby_edgesuid.astype('int64'))].copy()
    # We group the data by date and sum the total_trip_count. Each edge has a total_trip_count per day but and for each day we want to sum the trip counts over both edges, since they will then be compared 
    # to the data from the counter
    device_edges['date'] = pd.to_datetime(device_edges['date'])
    device_edges_aggregated = device_edges.groupby('date').sum()['strava_trips'].reset_index()
    return device_edges_aggregated

def combine_counter_and_strava(device_name,strava_dataset,gdf_bike_counters):
    # We load the counter history
    counter_history_daily = loading_and_grouping_counter_history(device_name,gdf_bike_counters)
    # We load the strava data for the nearby edges
    device_edges_aggregated = aggregating_nearby_strava_edges(device_name,strava_dataset,gdf_bike_counters)
    # We merge the two dataframes on the date column
    combined_data = counter_history_daily.merge(device_edges_aggregated, on='date',how = 'left')
    return combined_data

def add_weather_data_to_combined(combined_data,weather_data):
    # We merge the combined data with the weather data

    combined_data = combined_data.merge(weather_data, on='date',how = 'left')
    return combined_data


def remove_weekends_from_comined_datasets(total_data):
    # We remove the weekends from the data
    total_data['weekday']  = total_data['date'].apply(lambda x: x.isoweekday()) #Return the day of the week as an integer, where Monday is 1 and Sunday is 7.
    total_data_no_weekends = total_data.loc[~total_data['weekday'].isin([6,7])]
    return total_data_no_weekends

def remove_NaNs_and_zeros_from_datasets(total_data,counter_bool: bool = True):
    if counter_bool:
        total_data_nozeros = total_data.loc[(total_data['counter_trips'] != 0)].reset_index(drop = True).copy(deep = True)
        total_data_nozeros_noNaNs = total_data_nozeros.dropna(subset = ['counter_trips','strava_trips'],axis = 0).reset_index(drop = True).copy(deep = True)

    else:
        total_data_nozeros_noNaNs = total_data.dropna(subset = ['strava_trips'],axis = 0)


    return total_data_nozeros_noNaNs