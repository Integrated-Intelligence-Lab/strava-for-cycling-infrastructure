
from tqdm import tqdm
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
from shapely import wkt

from dotenv import load_dotenv
import os

load_dotenv()
import numpy as np
from src.loading_datasets import loading_bike_counters

DATA_PATH = os.getenv("DATA_PATH")

FULL_PAGE_WIDTH_CM = os.getenv("FULL_PAGE_WIDTH_CM")
FULL_PAGE_LENGTH_CM = os.getenv("FULL_PAGE_LENGTH_CM")

print(f"== Loading in Files == ")
strava_edge_total = pd.read_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total.csv")
strava_edge_shape = gpd.read_file(f'{DATA_PATH}/strava_edge_data/strava_edges.shp')

bike_counters_gpd = loading_bike_counters()

bcwl_vor_vertices = np.load(f'{DATA_PATH}/brussels/voronoi_vertices_counterbased.npy', allow_pickle=True)
for row in tqdm(bike_counters_gpd.iterrows(),desc = "Assosciating tessalated region to strava edges"):
    #Plot the regions    
    region = eval(row[1]["voronoi_region"])
    polygon = bcwl_vor_vertices[region]
    geometrical_polygon = Polygon(polygon)

    edges_in_polygon = strava_edge_total[strava_edge_total['edge_uid'].isin(
        strava_edge_shape[
            strava_edge_shape.geometry.intersects(geometrical_polygon)
        ]['edgeUID']
    )]
    strava_edge_total.loc[edges_in_polygon.index, 'counter_region_index'] = int(row[0])

    strava_edge_total.to_csv(f"{DATA_PATH}/strava_edge_data/strava_edge_total_locationid.csv",index = False)

print("== Sucessfuly saved strava_total with location ids ==")
