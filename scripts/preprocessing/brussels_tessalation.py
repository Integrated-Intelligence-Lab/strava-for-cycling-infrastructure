# %%
from scipy.spatial import Voronoi
import numpy as np
from src.geospatial_utilities import voronoi_finite_polygons_2d
from src.loading_datasets import loading_bike_counters, loading_brussels_shape
import os
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")
import matplotlib.pyplot as plt
from src.plotting_utilities import cm_to_inches
from shapely.geometry import Polygon
TEXT_WIDTH_CM = float(os.getenv("FULL_PAGE_WIDTH_CM"))
TEXT_HEIGHT_CM = float(os.getenv("FULL_PAGE_LENGTH_CM"))

bike_counters_gpd = loading_bike_counters()

#We create an array containing the coordinates of the counters. From this list we can create the Voronoi tessalation of the space
bike_counters_locations_array =  np.array(bike_counters_gpd.geometry.apply(lambda geom: [geom.x, geom.y] if geom else [None, None]).tolist())
bike_counters_locations_array_vor = Voronoi(bike_counters_locations_array)

bcwl_vor_regions, bcwl_vor_vertices = voronoi_finite_polygons_2d(bike_counters_locations_array_vor)

np.save(f'{DATA_PATH}/brussels/voronoi_vertices_counterbased.npy',bcwl_vor_vertices)
bike_counters_gpd["voronoi_region"] = bcwl_vor_regions

# Vertices is one long list of the coordinates of the vertices between voronoi regions. The bcwl_vor_regions contains, for each region,
# a list of indices used to index bcwl_vor_vertices in order to draw the outlines of the region. The order of regions is the same as the 
# order of the bike counters when iterating through the bike_counters_with_locations dataframe.

# %%
brussels_region_shape, _ = loading_brussels_shape()

minx, miny, maxx, maxy = brussels_region_shape.total_bounds


fig_voronoi_brussels, ax_voronoi_brussels = plt.subplots(figsize = (cm_to_inches(TEXT_WIDTH_CM/3),cm_to_inches(TEXT_WIDTH_CM/3)), dpi = 400)
_ = brussels_region_shape.plot(ax=ax_voronoi_brussels, color='lightgrey', edgecolor='black',linewidth = 0.1) # Brussels municipalities

for row in bike_counters_gpd.iterrows():
    #Plot the regions    
    region = row[1]["voronoi_region"]
    polygon = bcwl_vor_vertices[region]
    geometrical_polygon = Polygon(polygon)
    #Plot the counters
    ax_voronoi_brussels.plot(*geometrical_polygon.exterior.xy,color = 'red',linewidth = 0.5)
    ax_voronoi_brussels.fill(*zip(*polygon), alpha=0.4,color = 'grey')

    point = row[1]["geometry"]
    ax_voronoi_brussels.plot(point.coords[0][0], point.coords[0][1], 'o', color='blue',markersize =0.5)  # Bike counter locations
    center_polygon = geometrical_polygon.centroid

    ax_voronoi_brussels.text(point.coords[0][0], point.coords[0][1], row[1]["id"], fontsize=4, ha='right')

_ = ax_voronoi_brussels.set_xlim(minx-150,maxx+150)
_ = ax_voronoi_brussels.set_ylim(miny-150,maxy+150)

_ = ax_voronoi_brussels.set_xticks([])
_ = ax_voronoi_brussels.set_yticks([])

_ = [i.set_linewidth(0.1) for i in ax_voronoi_brussels.spines.values()]

fig_voronoi_brussels.savefig(f'{SAVE_PATH}/figures/main/figure_1_voronoi_regions_bike_counters.pdf', dpi=400, bbox_inches='tight')

