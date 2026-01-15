# %%
from dotenv import load_dotenv
import os
import json
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

import geopandas as gpd

# %%
## Load the Strava edge data
# We select the 2024 data as it is the most recent one but it should not vary from year to year. The basemap of the strava files is the same.
strava_edge_data_2024_s1_shape = gpd.read_file(f'{DATA_PATH}/strava_edge_data/2024/strava_edges_2024_s1.shp').to_crs('epsg:31370')
strava_edge_data_2024_s1_shape_union = strava_edge_data_2024_s1_shape.geometry.unary_union

# %%
non_union_gdf = strava_edge_data_2024_s1_shape
non_union_gdf.to_file(f'{DATA_PATH}/strava_edge_data/strava_edges.shp')

# Convert the unary_union geometry to a GeoDataFrame to save as shapefile
union_gdf = gpd.GeoDataFrame(geometry=[strava_edge_data_2024_s1_shape_union], crs='epsg:31370')
union_gdf.to_file(f'{DATA_PATH}/strava_edge_data/strava_edges_union.shp')
