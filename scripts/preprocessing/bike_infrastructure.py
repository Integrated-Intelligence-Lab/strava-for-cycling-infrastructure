
import geopandas as gpd
import pandas as pd
from src.loading_datasets import loading_brussels_shape, loading_brussels_cycling_network
from dotenv import load_dotenv
import os
import json
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
    
brussels_region_shape,_ = loading_brussels_shape()
bike_infra = loading_brussels_cycling_network()


# 1. Filter out infrastructure outside Brussels
bike_infra['in_brussels'] = None
bike_infra['in_brussels'] = bike_infra['geometry'].apply(lambda x: x.within(brussels_region_shape.geometry))

bike_infra = bike_infra[bike_infra['in_brussels']]

# 2. Code the types of infrastructure
bike_infra["type_coded"] = None
bike_infra.type_nl = pd.Categorical(bike_infra.type_nl)
bike_infra["type_coded"]  = bike_infra.type_nl.cat.codes
with open(f'{DATA_PATH}/bike_infrastructure/separated_infra_type.json') as f:
    separated_infrastructure_types = json.load(f)["Nl"]

with open(f'{DATA_PATH}/bike_infrastructure/infra_nl_eng_translation.json') as f:
    infrastructure_types_translated = json.load(f)


# 3. Subselect only the separated infrastructure types
separated_bike_infra = bike_infra.loc[bike_infra["type_nl"].isin(separated_infrastructure_types)].copy()

# 4. Translate the types to English
## Apperently, even if you select a subsample above the category still contains all the original categories. 
## So we need to translate all the categories, not only the subselected ones
separated_bike_infra['type_eng'] = separated_bike_infra['type_nl'].apply(lambda x: infrastructure_types_translated[x])

separated_bike_infra.to_csv(f"{DATA_PATH}/bike_infrastructure/separated_bike_infra.csv",index = False)
print("Separated bike infrastructure data saved.")