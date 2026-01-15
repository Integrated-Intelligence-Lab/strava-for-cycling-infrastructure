# %%
import geopandas as gpd
from shapely import wkt

from src.geospatial_utilities import find_closest_segment
from src.plotting_utilities import cm_to_inches
import math as math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os
import json
load_dotenv()

SAVE_PATH = os.getenv("RESULTS_PATH")
DATA_PATH = os.getenv("DATA_PATH")

FULL_PAGE_WIDTH_CM = os.getenv("FULL_PAGE_WIDTH_CM")
FULL_PAGE_LENGTH_CM = os.getenv("FULL_PAGE_LENGTH_CM")
# %%

## Load the bike counter data
bike_counters = gpd.read_file(f"{DATA_PATH}/bike_counters/raw/_counters_metadata.csv")
with open(f"{DATA_PATH}/bike_counters/_bike_counters_IDs.json") as f:
    bike_counters_IDs = json.load(f)

"""
Loading the geometry information from the file of the BikeCounters. 
The geometry is given in Lambert72 (crs = epsg:31370) ((so we convert it to Longitude and latitudes (crs = epsg:4326).))
"""
bike_counters_gpd = gpd.GeoDataFrame(
    data = bike_counters,
    geometry= bike_counters['geom'].apply(wkt.loads),crs = 'epsg:31370')

## Load the Strava edge data
strava_edges_shape = gpd.read_file(f'{DATA_PATH}/strava_edge_data/strava_edges.shp')
strava_edges_shape_union = gpd.read_file(f'{DATA_PATH}/strava_edge_data/strava_edges_union.shp')

# %% VERSION 1: Find the closest edgeUID to each counter based on projection to the combined line

# Iterate over each point
for idx,row in bike_counters_gpd.iterrows():
    point = row.geometry
    print(f"Iteration: {idx}")
    # Find the closest point on the combined line
    distance_s1 = strava_edges_shape_union.project(point)
    closest_point_s1 = strava_edges_shape_union.interpolate(distance_s1)
    
    # Find the closest line segment that contains the closest point
    closest_segment_idx_s1 = find_closest_segment(closest_point_s1, strava_edges_shape)
    
   
    bike_counters_gpd.at[idx,"edgeUID closest"] = strava_edges_shape.at[closest_segment_idx_s1,"edgeUID"]
    
    print(bike_counters_gpd.at[idx,"edgeUID closest"])

#We save this first iteration of the bike counters with the closest edgeUID
bike_counters_gpd.to_csv(f"{DATA_PATH}/bike_counters/counters_with_closestedge_v1.csv",index = False)
# %% VERSION 2: Manually verify and correct the closest edgeUIDs for each counter

#Some edges selected by the algorithm above have been changed as well as this was not the correct edges after further analysis. 
#The v_2 file of the counters with closest edges contain a column "edgeUID checked" which corresponds to the human assosciated ground truth, together with colums "Parallel edgeUID" 
#and "2nd Parallel edgeUID" which are parallell edges designated by humans as being also overlapping with the counter.
#The column "edgeUID" closest is the one selected by the algorithm above.

bike_counters_gpd = pd.read_csv(f"{DATA_PATH}/bike_counters/counters_with_closestedge_v2.csv")

bike_counters_gpd = gpd.GeoDataFrame(
    data = bike_counters_gpd,
    geometry= bike_counters_gpd['geom'].apply(wkt.loads),crs = 'epsg:31370')

figedge,axedge = plt.subplots(2,9,figsize = (cm_to_inches(float(FULL_PAGE_WIDTH_CM)),cm_to_inches(float(FULL_PAGE_LENGTH_CM)/8)),dpi = 325,
                            gridspec_kw = {'height_ratios': [1,1] ,'width_ratios': [1,1,1,1,1,1,1,1,1]},constrained_layout = True)


for idxk,row in bike_counters_gpd.iterrows():
    ax = axedge[math.floor(idxk/9),idxk%9]
    bike_counters_gpd.iloc[idxk:idxk+1].plot(ax =ax,color = 'black',alpha = 0.5,markersize = 2,aspect = None)

    #Originally Selected Edge
    edgeuid_og_s1 = [int(x) for x in row[['edgeUID checked']].unique() if not math.isnan(float(x))]
    strava_edges_shape.loc[strava_edges_shape['edgeUID'].isin(edgeuid_og_s1)].plot(ax = axedge[math.floor(idxk/9),idxk%9],color = 'green',aspect = None,linewidth = 0.5)
    #Parallel Edges manually added
    edgeuid_s1 = [int(x) for x in row[['Parallel edgeUID','2nd Parallel edgeUID']].unique() if not math.isnan(float(x))]
    
   
    if len(edgeuid_s1) != 0:  
        strava_edges_shape.loc[strava_edges_shape['edgeUID'].isin(edgeuid_s1)].plot(ax = ax,color = 'orange',aspect = None,linewidth = 0.5)
    
    ax.set_yticks([])
    ax.set_xticks([])

    axedge[math.floor(idxk/9),idxk%9].text(0, 1.15,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes,s = r"$\mathbf{" + row['device_name'] + "}$",fontsize = 5)
    ax.spines['top'].set_linewidth(0.2)
    ax.spines['right'].set_linewidth(0.2)
    ax.spines['bottom'].set_linewidth(0.2)
    ax.spines['left'].set_linewidth(0.2)

figedge.savefig(f"{SAVE_PATH}/figures/bike_counters_with_verified_edges_v2.pdf",dpi = 325,)

# %% VERSION 3: Add VORONOI regions

bcwl_vor_regions = np.load(f'{DATA_PATH}/brussels/voronoi_regions_counterbased.npy',allow_pickle=True)
bike_counters_gpd["voronoi_region"] = bcwl_vor_regions

bike_counters_gpd.to_csv(f"{DATA_PATH}/bike_counters/counters_with_closestedge_v3.csv",index = False)

