import geopandas as gpd
from shapely import wkt
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

def loading_bike_counters(original:bool = False):

    if original:
        ## Load the bike counter data
        bike_counters = gpd.read_file(f"{DATA_PATH}/bike_counters/raw/_counters_metadata.csv")
    else:
        bike_counters =  pd.read_csv(f"{DATA_PATH}/bike_counters/counters_with_closestedge_v3.csv")

    """
    Loading the geometry information from the file of the BikeCounters. 
    The geometry is given in Lambert72 (crs = epsg:31370) ((so we convert it to Longitude and latitudes (crs = epsg:4326).))
    """
    bike_counters_gpd = gpd.GeoDataFrame(
        data = bike_counters,
        geometry= bike_counters['geom'].apply(wkt.loads),crs = 'epsg:31370')

    return bike_counters_gpd

def loading_brussels_shape():
    brussels_region_shape = gpd.read_file(f"{DATA_PATH}/brussels/UrbAdm_REGION.shp")
    brussels_region_shape.to_crs(epsg=31370, inplace=True)

    brussels_municipal_shape = gpd.read_file(f"{DATA_PATH}/brussels/UrbAdm_MUNICIPALITY.shp")
    brussels_municipal_shape.to_crs(epsg=31370, inplace=True)
    return brussels_region_shape, brussels_municipal_shape

def loading_brussels_cycling_network():
    bike_infrastructure_data = gpd.read_file(f"{DATA_PATH}/bike_infrastructure/infra_lambert72.csv")
    bike_infrastructure_data_with_locations = gpd.GeoDataFrame(
        data = bike_infrastructure_data,
        geometry= bike_infrastructure_data['geom'].apply(wkt.loads),crs = 'epsg:31370')

    return bike_infrastructure_data_with_locations

def loading_separated_bike_infrastructure():
    separated_bike_infra = pd.read_csv(f"{DATA_PATH}/bike_infrastructure/separated_bike_infra.csv")
    separated_bike_infrastructure_with_locations = gpd.GeoDataFrame(
        data = separated_bike_infra,
        geometry= separated_bike_infra['geom'].apply(wkt.loads),crs = 'epsg:31370')
    
    return separated_bike_infrastructure_with_locations