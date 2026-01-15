# %%
import pandas as pd

from dotenv import load_dotenv
import os
from tqdm import tqdm
load_dotenv()
SAVE_PATH = os.getenv("RESULTS_PATH")
DATA_PATH = os.getenv("DATA_PATH")

# %%

strava_edge_total = pd.DataFrame({})

for year in tqdm([2019,2020,2021,2022,2023,2024],desc= "Merging Strava datasets"):

    strava_edge_data_xxxx_s1 = pd.read_csv(f'{DATA_PATH}/strava_edge_data/{year}/strava_edges_{year}_s1.csv')
    strava_edge_data_xxxx_s2 = pd.read_csv(f'{DATA_PATH}/strava_edge_data/{year}/strava_edges_{year}_s2.csv')
    strava_edge_data_xxxx_s3 = pd.read_csv(f'{DATA_PATH}/strava_edge_data/{year}/strava_edges_{year}_s3.csv')
    if year < 2024:
        strava_edge_data_xxxx_s4 = pd.read_csv(f'{DATA_PATH}/strava_edge_data/{year}/strava_edges_{year}_s4.csv')
        temp_xxxx = pd.concat([strava_edge_data_xxxx_s1,strava_edge_data_xxxx_s2,strava_edge_data_xxxx_s3,strava_edge_data_xxxx_s4])
        del strava_edge_data_xxxx_s1,strava_edge_data_xxxx_s2,strava_edge_data_xxxx_s3,strava_edge_data_xxxx_s4
    else:
        temp_xxxx = pd.concat([strava_edge_data_xxxx_s1,strava_edge_data_xxxx_s2,strava_edge_data_xxxx_s3])
        del strava_edge_data_xxxx_s1,strava_edge_data_xxxx_s2,strava_edge_data_xxxx_s3

    strava_edge_data_xxxx_grouped = temp_xxxx.groupby(['edge_uid','date'])[['total_trip_count', 'forward_commute_trip_count','reverse_commute_trip_count','forward_leisure_trip_count','reverse_leisure_trip_count','ride_count']].sum().copy(deep=True)
    del temp_xxxx
    strava_edge_data_xxxx_grouped  = strava_edge_data_xxxx_grouped.reset_index()
    strava_edge_total = pd.concat([strava_edge_total,strava_edge_data_xxxx_grouped])
    
# %%
strava_edge_total['date'] = pd.to_datetime(strava_edge_total['date'])
strava_edge_total.sort_values(by = "date", ascending=True,inplace = True)
strava_edge_total.rename(columns={"total_trip_count": "strava_trips"}, inplace=True)
strava_edge_total.to_csv(f'{DATA_PATH}/strava_edge_data/strava_edge_total.csv',index = False)
print("Succesfully saved merged dataset")