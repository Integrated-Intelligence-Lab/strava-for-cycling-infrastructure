
# %% IMPORTS
import pandas as pd
import geopandas as gpd


from shapely import wkt
from shapely.geometry import Point

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import json
load_dotenv()
SAVE_PATH = os.getenv("RESULTS_PATH")
DATA_PATH = os.getenv("DATA_PATH")

from src.loading_datasets import loading_brussels_shape

if __name__ == "__main__":

    # We load in the kmi file, which contains the weather data for all stations. We then first convert the (lat,long) to (long,lat) in order to have the appropriate compatible geometry with the brussels region shape file.
    # We then identify the weather station (Uccle) that is within the brussels capital region
    brussels_region, _ = loading_brussels_shape()

    kmi_total = pd.read_csv(f'{DATA_PATH}/weather/aws_1day.csv')
    # We need to convert the 'the_geom' column from WKT to a geometry object
    weather_gpd = gpd.GeoDataFrame(
        data = kmi_total,
        geometry= kmi_total['the_geom'].apply(wkt.loads),crs = 'epsg:4236')

    weather_gpd["date"] =  pd.to_datetime(weather_gpd['timestamp'])

    # We need to invert the coordinates from (latitude, longitude) to (longitude, latitude)
    for point in weather_gpd['the_geom'].unique():
        coords = point.replace('POINT (', '').replace(')', '').split()
        new_point = Point(float(coords[1]), float(coords[0]))

        weather_gpd.loc[weather_gpd['the_geom'] == point, 'geometry'] = new_point

    weather_gpd['geometry'] = weather_gpd['geometry'].to_crs('epsg:31370')

    # Find the one point that is within the Brussels region shape
    for point in weather_gpd['geometry'].unique():
        if point.within(brussels_region.geometry.iloc[0]):
            point_in_brussels = point
    # %%
    # The weather data in Brussels has some missing data (especially for the wind measurements). 
    # To get an estimate for this missing date we take the average of the three weather stations most closely located around Brussels

    # Get the Brussels region polygon (assuming there is only one)
    brussels_poly = brussels_region.geometry.iloc[0]

    # Extract vertices from the polygon's exterior boundary
    vertices_coords = list(brussels_poly.exterior.coords)
    vertex_points = [point for point in weather_gpd['geometry'].unique()]
    # Use the provided point_in_brussels as the reference point
    reference = point_in_brussels

    # Exclude any vertex that exactly equals the reference point
    vertex_points_filtered = [pt for pt in vertex_points if not pt.equals(reference)]

    # Calculate distances from each vertex to the reference point
    distance_list = [(pt, pt.distance(reference)) for pt in vertex_points_filtered]

    # Sort the list by distance and get the three closest points
    three_closest = sorted(distance_list, key=lambda item: item[1])[:3]
    three_closest_points_geometry = [point[0] for point in three_closest]

    print("The three closest vertices to the reference point are:")
    for pt, dist in three_closest:
        print(f"Point: {pt}, Distance (km): {dist/1000:.2f}")

    # %%# Plot to visualize the points

    fig_weather,ax_weather = plt.subplots(figsize=(5,2.5),dpi=300)

    brussels_region.geometry.boundary.plot(ax = ax_weather,color = None, edgecolor = 'black',alpha = 0.5, linewidth = 0.5)
    weather_gpd.plot(ax = ax_weather, color = 'black', markersize = 1,label = 'Weather stations')
    weather_gpd[weather_gpd['geometry'] == point_in_brussels].plot(ax = ax_weather, color = 'red', markersize = 1, label = 'Weather station\n in Brussels')
    weather_gpd[weather_gpd['geometry'].isin(three_closest_points_geometry)].plot(ax = ax_weather, color = 'cyan', markersize = 1, label = 'Closest weather\n stations')

    _ = ax_weather.set_xticks([])
    _ = ax_weather.set_yticks([])

    ax_weather.legend(loc = 'upper right', fontsize = 6, frameon = False, markerscale = 1,bbox_to_anchor=(1.47, 1.0))

    fig_weather.savefig(f'{SAVE_PATH}/figures/weather_stations_brussels_location.pdf', dpi=300, bbox_inches='tight')

    # %%

    weather_data_in_brussels = weather_gpd.loc[weather_gpd['geometry'] == point_in_brussels].reset_index(drop=True)
    weather_data_in_brussels = weather_data_in_brussels[['date','temp_avg','precip_quantity','wind_speed_10m']]

    weather_data_around_brussels = weather_gpd[weather_gpd['geometry'].isin(three_closest_points_geometry)].reset_index(drop=True)
    weather_data_around_brussels = weather_data_around_brussels[['date','temp_avg','precip_quantity','wind_speed_10m']]

    weather_data_around_brussels = weather_data_around_brussels.groupby('date', as_index=False).mean().reset_index(drop=True) 

    weather_data_brussels_final = pd.merge(weather_data_in_brussels, weather_data_around_brussels, on='date', how='outer',suffixes=('_in_brussels', '_around_brussels'))
    weather_data_brussels_final = weather_data_brussels_final.rename(columns={
        'temp_avg_in_brussels': 'temp_avg',
        'precip_quantity_in_brussels': 'precip_quantity',
        'wind_speed_10m_around_brussels': 'wind_speed_10m'
    })

    weather_data_brussels_final = weather_data_brussels_final[['date','temp_avg','precip_quantity','wind_speed_10m']]

    weather_data_brussels_final.to_csv(f'{DATA_PATH}/weather/weather_data_complete_vnew.csv',index = False)

