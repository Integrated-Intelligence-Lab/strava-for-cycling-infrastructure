# %% 
from dotenv import load_dotenv
import geopandas as gpd

import os
import json
load_dotenv()

SAVE_PATH = os.getenv("RESULTS_PATH")
DATA_PATH = os.getenv("DATA_PATH")



## Load the bike counter data
bike_counters = gpd.read_file(f"{DATA_PATH}/bike_counters/_counters_metadata.csv")

bike_counters_IDs = bike_counters["device_name"].unique().tolist()

with open(f"{DATA_PATH}/bike_counters/_bike_counters_IDs.json", "w") as f:
    json.dump(bike_counters_IDs, f)