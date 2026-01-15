import geopandas as gpd
import shapely
from tqdm import tqdm
import pandas as pd

from shapely import wkt
from shapely.geometry import box
from shapely.strtree import STRtree
import math

import numpy as np

from dotenv import load_dotenv
import os
from src.geospatial_utilities import bearing, angle_diff,create_polysplit_region
from src.loading_datasets import loading_brussels_shape,loading_separated_bike_infrastructure
from src.plotting_utilities import cm_to_inches,setup_plotting
from src.aux_utilities import w_logistic, w_clipped_gauss

import matplotlib.pyplot as plt
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")
TOTAL_PAGE_WIDTH_CM = float(os.getenv("FULL_PAGE_WIDTH_CM"))
TOTAL_PAGE_LENGTH_CM = float(os.getenv("FULL_PAGE_LENGTH_CM"))



# Hyperparameters of the search
buffer_values = [1,2,4,8,12,16,20]
min_ratio_values = [0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
max_angle_values = [2,4,6,8,10,12,16,20,25,30,35,40,50,60,70,80]

def sample_points(line, spacing=10):
    pts = []
    for part in line.geoms if line.geom_type == 'MultiLineString' else [line]:
        d = 0
        while d < part.length:
            pts.append(part.interpolate(d))
            d += spacing
    return pts

def _sample_points_line(geom, spacing=10.0):
    """Yield points every `spacing` m along (Multi)LineString."""
    if geom.is_empty:
        return
    if geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            yield from _sample_points_line(part, spacing)
    elif geom.geom_type == "LineString":
        d = 0.0
        L = geom.length
        while d <= L:
            yield geom.interpolate(d)
            d += spacing

def fuzzy_scores(sel, infra, sigma=7, spacing=10,weight_function='gaussian'):
    #
    if weight_function == 'logistic':
        def w(d):
            return w_logistic(d, sigma=sigma, k=2.0)
    elif weight_function == 'gaussian':
        def w(d): 
            return math.exp(-(d*d)/(2*sigma*sigma))
    elif weight_function == 'clipped_gaussian':
        def w(d):
            return w_clipped_gauss(d, sigma=sigma, tau=2.0)
        
    pts_sel   = sample_points(sel, spacing)
    pts_inf   = sample_points(infra, spacing)
    prec_list = []
    rec_list  = []
    for p in pts_sel:
        prec_list.append(w(p.distance(infra)))
    for p in pts_inf:
        rec_list.append(w(p.distance(sel)))

    prec = np.mean(prec_list)
    rec  = np.mean(rec_list)
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {
        'precision_pct': prec,
        'recall_pct': rec,
        'f1_score': f1
    }

def scoring_edges_infrastructure(infra_gdf,strava_shape,
                                 buffer_values, min_ratio,max_angle,sigma_value,
                                 tol = 1):

    # 1 row per LineString → bearing & STRtree stay simple
    infra_lines = (
        infra_gdf
        .explode(index_parts=False, ignore_index=True)       # split MultiLineStrings
        .loc[lambda df: df.geometry.type == "LineString"]    # keep only lines
    )

    # keep a list/array of the geometries that went into the STRtree
    infra_geoms = infra_lines.geometry.to_list()      # same order!
    infra_tree  = STRtree(infra_geoms,node_capacity=50)
    infra_union = infra_gdf.unary_union

    # Copy of the edges GeoDataFrame to avoid modifying the original
    edges_gdf_total = strava_shape.copy()

    edge_analysis_results = pd.DataFrame(columns=['buffer_value_m','min_ratio','max_angle_deg',
                                                'kilometers_edges_sel'
                                                ,'precision_pct','recall_pct','f1_score'])

    for min_ratio in tqdm(min_ratio_values, desc="min_ratio"):
        for max_angle in max_angle_values:

            # fresh copy for this parameter set
            edges_gdf = edges_gdf_total.copy()
            
            # -------------------------------------------------------------
            # 1A.  FLAG EDGES THAT POSSESS PROTECTED INFRA
            # -------------------------------------------------------------
            for idx, row in edges_gdf.iterrows():

                edge_geom  = row.geometry
            
                edge_bear  = bearing(edge_geom)

                # candidates within the WIDEST corridor
                maxi_buf   = max(buffer_values)
                cand_idx   = infra_tree.query(edge_geom.buffer(maxi_buf + tol))
                

                for b in buffer_values:
                    buf_geom = row[f'buffer_{b}m']

                    # quick reject
                    if not any(infra_geoms[i].buffer(tol).intersects(buf_geom) for i in cand_idx): #SYMMETRIC IF WE BUFFER?
                        edges_gdf.at[idx, f'separated_bike_infrastructure_{b}m'] = False
                        continue


                    ok = False
                    for i in cand_idx:
                        infra_geom = infra_geoms[i]
                        
                        overlap_len = edge_geom.intersection( 
                            infra_geom.buffer(b + tol)   
                            ).length

                        if overlap_len/edge_geom.length < min_ratio:                  
                            continue
                        if angle_diff(edge_bear, bearing(infra_geom)) > max_angle:
                            continue
                        ok = True
                        break

                    edges_gdf.at[idx, f'separated_bike_infrastructure_{b}m'] = ok

            # -------------------------------------------------------------
            # 1B.  GLOBAL METRICS PER BUFFER VALUE
            # -------------------------------------------------------------
            for b in buffer_values:
                edge_infra = edges_gdf[edges_gdf[f'separated_bike_infrastructure_{b}m']]

                sel_union     = edge_infra.geometry.unary_union        # dissolve

                precision_pct_gauss,recall_pct_gauss,f1_gauss = fuzzy_scores(sel_union, infra_union, sigma=sigma_value, spacing=1,weight_function='gaussian')
                precision_pct_clippedgauss,recall_pct_clippedgauss,f1_clippedgauss = fuzzy_scores(sel_union, infra_union, sigma=sigma_value, spacing=1,weight_function='clipped_gauss')

                edge_analysis_results.loc[len(edge_analysis_results)]  = {
                    'buffer_value_m'        : b,
                    'min_ratio'             : min_ratio,
                    'max_angle_deg'         : max_angle,
                    'kilometers_edges_sel'  : sel_union.length / 1000 if sel_union else 0,
                    'precision_pct_gauss'         : precision_pct_gauss,
                    'recall_pct_gauss'            : recall_pct_gauss,
                    'f1_score_gauss'              : f1_gauss,
                    'precision_pct_clippedgauss'         : precision_pct_clippedgauss,
                    'recall_pct_clippedgauss'            : recall_pct_clippedgauss,
                    'f1_score_clippedgauss'              : f1_clippedgauss
                }

    edge_analysis_results.to_csv(f'{SAVE_PATH}/datasets/edge_infra_match_sigma{sigma_value}.csv', index=False)

# This code is the same implementation as above. However we don't score the selection but simply look at the selection based on the optimal hyperparameters
def matching_edges_infra_with_hyperparameters(infra_gdf,edges_gdf,
                          final_buffer_value,final_min_ratio,final_max_angle,
                          poly = None, iteration: int = 1,display_progress = False,tol = 1):


    # 1 row per LineString → bearing & STRtree stay simple
    infra_lines = (
        infra_gdf
        .explode(index_parts=False, ignore_index=True)       # split MultiLineStrings
        .loc[lambda df: df.geometry.type == "LineString"]    # keep only lines
    )

    # keep a list/array of the geometries that went into the STRtree
    infra_geoms = infra_lines.geometry.to_list()      # same order!
    infra_tree  = STRtree(infra_geoms,node_capacity=50)

    edges_gdf_total = edges_gdf.copy()

    edges_gdf_total[f"buffer_{final_buffer_value}m"] = edges_gdf_total.buffer(final_buffer_value,cap_style = "flat")

    for idx, row in tqdm(edges_gdf_total.iterrows(), disable=not display_progress,desc=f"Matching edges with infra - iteration {iteration}"):

        edge_geom  = row.geometry
            
        edge_bear  = bearing(edge_geom)

        # candidates within the WIDEST corridor
        maxi_buf   = max(buffer_values)

        if poly is None:
            cand_idx = infra_tree.query(edge_geom.buffer(maxi_buf + tol))
        else:
            cand_idx = infra_tree.query(poly, predicate='intersects')
                
        buf_geom = row[f'buffer_{final_buffer_value}m']

        # quick reject
        if not any(infra_geoms[i].buffer(tol).intersects(buf_geom) for i in cand_idx): #SYMMETRIC IF WE BUFFER?
            edges_gdf_total.at[idx, f'sep_it_{iteration}'] = False
            continue


        ok = False
        for i in cand_idx:
            infra_geom = infra_geoms[i]
                        

            overlap_len = edge_geom.intersection( 
                    infra_geom.buffer(final_buffer_value + tol)   
                    ).length
            if overlap_len/edge_geom.length < final_min_ratio:                  
                continue
            if angle_diff(edge_bear, bearing(infra_geom)) > final_max_angle:
                continue
            ok = True
            break

        edges_gdf_total.at[idx, f'sep_it_{iteration}'] = ok

    edges_gdf_total = edges_gdf_total.drop(f'buffer_{final_buffer_value}m', axis=1)
    return edges_gdf_total


def expand_tile_bbox_by_farthest_outside(
    tile_geom,                # shapely Polygon (your tile)
    infra_intersecting_gdf,   # ONLY infra that intersects tile
    base=0.0,                 # minimum expansion (m)
    margin=10.0,              # safety slack (m)
    spacing=10.0,             # sampling step (m) along outside infra
):
    """
    Return an expanded *rectangle* (bbox) around `tile_geom` by XX metres, where
    XX is determined from how far the *outside* parts of intersecting infra extend.
    """
    # 1) collect distances from outside-infra points to the tile (edge)
    dists = []
    #for g in infra_intersecting_gdf.geometry:
    outside = infra_intersecting_gdf.difference(tile_geom)      # only the parts beyond the tile

        
    for p in _sample_points_line(outside, spacing=spacing):
        dists.append(p.distance(tile_geom))  # distance to the box (same as to edge)

    # 2) choose expansion XX
    if not dists:
        XX = base
    else:
        d_star = max(dists)
        XX = max(base, d_star + margin)


    # 3) expand the *bbox* (keeps it rectangular)
    minx, miny, maxx, maxy = tile_geom.bounds
    return box(minx - XX, miny - XX, maxx + XX, maxy + XX), XX

def evaluation_metrics_per_polygon_firstrun(infra_gdf, df_edges, sigma_value,region_shape):
    evaluation_metrics_per_polygon = []

    polys = create_polysplit_region(region=region_shape)
    
    sel_edges = df_edges[df_edges[f"sep_it_1"]]

    for i, poly in tqdm(enumerate(polys)):
        # Get the infrastructure part within the polygon
        infra_gdf_intersect = infra_gdf[infra_gdf.intersects(poly)]
        infra_eval = gpd.clip(infra_gdf_intersect, poly)
        if infra_gdf_intersect.empty:
            continue

        sel_gdf = sel_edges[sel_edges.intersects(poly)]
        sel_eval = gpd.clip(sel_gdf, poly)

        sel_eval = sel_eval.explode(index_parts=False, ignore_index=False) #To only contain Linestrings

        if sel_eval.empty:
            recall_pct = np.nan
            f1 = np.nan
            precision_pct = np.nan
            evaluation_metrics_per_polygon.append({'polygon_index': i,'recall_pct': recall_pct,'f1_score': f1, 'precision_pct': precision_pct,})
        else:
            metric = fuzzy_scores(sel_eval.unary_union, infra_eval.unary_union, sigma=sigma_value, spacing=1, weight_function='clipped_gaussian')
            evaluation_metrics_per_polygon.append({'polygon_index': i, 'recall_pct': metric['recall_pct'], 'f1_score': metric['f1_score'], 'precision_pct': metric['precision_pct']})

    df_evaluation_per_polygon = pd.DataFrame(evaluation_metrics_per_polygon)

    return df_evaluation_per_polygon

def assign_poly_to_infra(infrastructure_union: shapely.geometry.base.BaseGeometry,polys= None, ax=None):

    if polys is None:
        print("Poly set empty, creating new polygons.")
        polys = create_polysplit_region(region=brussels_region_shape)

    # Create GeoDataFrame 
    grid_gdf = gpd.GeoDataFrame({'geometry': polys}, crs=brussels_region_shape.crs)


    for poly in polys:
        if infrastructure_union.intersects(poly) & (infrastructure_union.intersection(poly).length > 10):
            grid_gdf.loc[grid_gdf.geometry == poly, 'infrastructure'] = True
            
        else:
            grid_gdf.loc[grid_gdf.geometry == poly, 'infrastructure'] = False

    return grid_gdf

def iterative_selection(infra_grid,edges_selected_df,
                        bike_infra, strava_edge_data,
                        number_of_iterations_per_sigma:int = 40,
                        threshold_value: float = 0.95,measure_to_consider: str = "f1_score",
                        sigma_list = None
                        ):

    bad_cells = infra_grid[(infra_grid[f'{measure_to_consider}_treshold'] == 0) & (infra_grid['infrastructure'])].copy()

    if sigma_list is None:
        raise NotImplementedError("Please provide a list of sigma values to use during the iterative selection.")
    

    for idx in tqdm(range(len(sigma_list))):
        NEW_SIGMA = sigma_list[idx]
        print(f"\n====SIGMA {NEW_SIGMA}====\n")
        iteration = 1
        setups_to_use = pd.read_csv(f"{SAVE_PATH}/datasets/strava_infrastructure_match/edge_infra_match_sigma{NEW_SIGMA}.csv")

        edges_selected_df[f'sep_it_SIGMA{NEW_SIGMA}'] =False

        infra_grid[f'recall_pct_SIGMA{NEW_SIGMA}'] = None
        infra_grid[f'f1_score_SIGMA{NEW_SIGMA}'] = None
        infra_grid[f'precision_pct_SIGMA{NEW_SIGMA}'] = None
        while not bad_cells.empty and iteration <= number_of_iterations_per_sigma:
            print(f"Iteration {iteration}, Bad cells remaining: {len(bad_cells)}")
            next_setup = setups_to_use['f1_score_clippedgauss'].nlargest(iteration).index[-1]
            next_setup_parameters = {
                'final_buffer_value': int(setups_to_use.loc[next_setup]['buffer_value_m'].item()),
                'final_min_ratio': setups_to_use.loc[next_setup]['min_ratio'].item(),
                'final_max_angle': setups_to_use.loc[next_setup]['max_angle_deg'].item()
            }

            for index, bad_cell in bad_cells.iterrows():
                infra = bike_infra[bike_infra.intersects(bad_cell.geometry)]
                infra_eval = gpd.clip(infra, bad_cell.geometry)

                edges = strava_edge_data[strava_edge_data.intersects(bad_cell.geometry)]
                edges_eval = edges.clip(bad_cell.geometry)
                
                edges_eval = edges_eval.explode(index_parts=False, ignore_index=False) #To only contain Linestrings

               # Run the edge infrastructure algorithm with the selected parameters
                temp = matching_edges_infra_with_hyperparameters(infra_eval, edges_eval, **next_setup_parameters,iteration = idx)

                if len(temp) == 0:
                    continue
                sel_edges = temp[temp[f"sep_it_{idx}"]]
                if len(sel_edges) == 0:
                    continue
                fuzzy_scores_result = fuzzy_scores(sel_edges.unary_union, infra_eval.unary_union, sigma=NEW_SIGMA, spacing=1, weight_function='clipped_gaussian')

                if fuzzy_scores_result[measure_to_consider] > threshold_value:
                    print(f"Improved cell : {index}")

                    bad_cells = bad_cells[bad_cells.geometry != bad_cell.geometry]

                    infra_grid.loc[index, f'precision_pct_SIGMA{NEW_SIGMA}'] = fuzzy_scores_result['precision_pct'].item()
                    infra_grid.loc[index, f'recall_pct_SIGMA{NEW_SIGMA}'] = fuzzy_scores_result['recall_pct'].item()
                    infra_grid.loc[index, f'f1_score_SIGMA{NEW_SIGMA}'] = fuzzy_scores_result['f1_score'].item()


                    edges_selected_df.loc[sel_edges.index, f'sep_it_SIGMA{NEW_SIGMA}'] = sel_edges[f'sep_it_{idx}'].astype(bool)
            iteration +=1
    return edges_selected_df, infra_grid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_values', type=int, nargs='+', default=[*range(4,21)], help='sigma value for fuzzy scoring')
    parser.add_argument('--rerun_scoring', action='store_true', help='Whether to rerun the scoring over the hyperparameter grid')
    parser.add_argument('--measure_to_consider',type = str,default = 'f1_score', help = 'measure to consider during the iterative selection')
    parser.add_argument('--threshold_value',type = float,default = 0.95, help = 'threshold value to use during evaluation')
    
    args = parser.parse_args()
    """
    Main function to run the matching between Strava edges and cycling infrastructure in Brussels.
    It first scores the matching based on a grid of hyperparameters (buffer size, minimum ratio, maximum angle)
    and then performs an iterative selection process to improve the matching in underperforming areas.
    """
    sigma_value = args.sigma_values
    brussels_region_shape,_ = loading_brussels_shape()
    bike_infra = loading_separated_bike_infrastructure()
    strava_edge_shape = gpd.read_file(f'{DATA_PATH}/strava_edge_data/strava_edges.shp').to_crs('epsg:31370')
    label_font, tick_font, title_font, legend_font = setup_plotting()
    
    #for buffer_value in buffer_values:
    #    strava_edge_shape[f"buffer_{buffer_value}m"] = strava_edge_shape.buffer(buffer_value,cap_style = "flat")

    TOL = 1 # positional-tolerance buffer (m)

    # This creates the CSV files with the scoring results based on the hyperparameter grid for each sigma value between 4 and 20
    if args.rerun_scoring:
        scoring_edges_infrastructure(bike_infra, strava_edge_shape,
                                    buffer_values, min_ratio_values, max_angle_values, sigma_value,
                                    tol = TOL)
    else:
        print("Skipping scoring step as per user request. Using saved files")
    
    # Once we have all the CSV files, we can analyze them in a notebook to find the optimal hyperparameters
    
    # a. We start with sigma = 4 
    first_edge_analysis_results = pd.read_csv(f'{SAVE_PATH}/datasets/strava_infrastructure_match/edge_infra_match_sigma4.csv')
    top_idx_clippedgauss = first_edge_analysis_results['f1_score_clippedgauss'].nlargest(1).index
    top_clippedgauss_parameters = {
    'final_buffer_value': first_edge_analysis_results.iloc[top_idx_clippedgauss.item()]['buffer_value_m'].item(),
    'final_min_ratio': first_edge_analysis_results.iloc[top_idx_clippedgauss.item()]['min_ratio'].item(),
    'final_max_angle': first_edge_analysis_results.iloc[top_idx_clippedgauss.item()]['max_angle_deg'].item()
    }
    # This iteration happens at the global level with sigma = 4
    initial_iteration_edges_infra = matching_edges_infra_with_hyperparameters(
        bike_infra, strava_edge_shape,
        **top_clippedgauss_parameters,
        display_progress=True,
        tol= TOL
    )
    
    infra_grid = assign_poly_to_infra(bike_infra.unary_union)
    df_eval_polygon = evaluation_metrics_per_polygon_firstrun(bike_infra,initial_iteration_edges_infra, sigma_value=4,region_shape=brussels_region_shape)

    infra_grid['recall_pct'] = df_eval_polygon.set_index('polygon_index')['recall_pct']
    infra_grid['f1_score'] = df_eval_polygon.set_index('polygon_index')['f1_score']
    infra_grid['precision_pct'] = df_eval_polygon.set_index('polygon_index')['precision_pct']

    infra_grid['f1_score_treshold'] = infra_grid['f1_score'].apply(lambda x: 1 if x >= args.threshold_value else 0)
    infra_grid['recall_pct_treshold'] = infra_grid['recall_pct'].apply(lambda x: 1 if x >= args.threshold_value else 0)
    infra_grid['precision_pct_treshold'] = infra_grid['precision_pct'].apply(lambda x: 1 if x >= args.threshold_value else 0)


    fig_initial_iteration, ax = plt.subplots(figsize=(cm_to_inches(TOTAL_PAGE_WIDTH_CM)/2, cm_to_inches(TOTAL_PAGE_WIDTH_CM)/2),dpi = 500,sharex=True,sharey=True,constrained_layout = True)

    ax.text(0,1.02,
        ha = 'left', va = 'bottom',
        s = f"Buffer: {top_clippedgauss_parameters['final_buffer_value']}m, Min Ratio: {top_clippedgauss_parameters['final_min_ratio']}, Max Angle: {top_clippedgauss_parameters['final_max_angle']}°",
        transform = ax.transAxes,
        fontdict = legend_font)

    bike_infra.plot(ax=ax, color='blue', linewidth=0.5, label='Infrastructure')
    strava_edge_shape.plot(ax=ax, color='green', linewidth=0.5, label='Strava Edges',alpha=0.1)

    initial_iteration_edges_infra[initial_iteration_edges_infra[f"sep_it_1"]].plot(ax=ax, color='red', linewidth=0.5, label='Selected Edges',alpha=0.5)


    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig_initial_iteration.savefig(f"{SAVE_PATH}/figures/strava_infra_matching_initial_iteration.pdf")
    initial_iteration_edges_infra.to_csv(f'{SAVE_PATH}/datasets/strava_edges_with_infra_initial_iteration.csv', index=False)
    infra_grid.to_csv(f'{SAVE_PATH}/datasets/infrastructure_grid_initial_run.csv')


    sigma_list = [4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20]
    final_iteration_edges_infra, infra_gridv2 = iterative_selection(
        infra_grid,initial_iteration_edges_infra,
        bike_infra = bike_infra, strava_edge_data = strava_edge_shape,
        number_of_iterations_per_sigma= 40,
        threshold_value= 0.95,measure_to_consider= args.measure_to_consider,
       sigma_list = sigma_list    
    )

    # Merge all sep_it_* columns into one boolean column: True if any is True
    sep_cols = final_iteration_edges_infra.filter(regex=r'^sep_it_').columns
    final_iteration_edges_infra['separated_bike_infrastructure_any'] = final_iteration_edges_infra[sep_cols].fillna(False).any(axis=1)

    """
    Manual inspection of the results showed that some edges where incorrectly marked as having separated bike infrastructure.
    """

    additional_true = [160227334,160227327,160227326,160227273,160227271,160227056,160227268,160228420,160193296,160193310,160364498,160364514,160364091,160364074,160364496,160380574,160380577,
                   160380583,160380584,160380587,160380586,160408902,160408872,160410405,160410404,160410403,160410408,160410372,160410377,160410411,160410409,160410413,160410407,
                   160226580,160226570,160226577,160226518,160226715,160226731,160226722,160226828,160226833,160226738,160226739,160226745,160226719,160226576,160226565,
                   160302119,160301782,160301781,160301777,160301776,160302553,160302101,160302102,160302112,160301916,160301780,160302436,160301773,160301774,160302438,160411046,
                   160411026,160411025,160411021,160411020,160411018,160411016,160411015,160410638,160410640,160410636,160410631,160410622,160410620,160410618,160381949,160381948,160381940,160381939,160411060,160381930,160411067,160411069,160411065,
                   160411206,160411063,160381889,160381928,160411048,
                   160177442,160177520,160177523,160177502,160191586,160191956,160191970,160195482,160195426,160363137,160230070,160299520,160320554,160320549,160177468,160191945,
                   160193288,160195498,160193998,160194004,160194003,160194002,160352325,160352321,160353553,160365932,
                   160242027,160242028,160177549,160177546,160177538,160177535,160177532,160177533,160175344,160177428,
                   160230444,160230450,160230550,160230565,160230569,160230567,160230573,160230571,160230577,160230575
                   ]

    additional_false = [160363182,160363184,160363183,160363089,160191898,160301783,160301784,160302590,160301753,160381943,160381942,160410619,160410634,160410635,160410621,160410626,160410624,160410627,160410623,160410632,
                        160381890,160381885,160381929,160411039,160410639,160381888,160381858,160410630,
                        160241780,160271743,160191524,160191525,160288896,160288895,160288897,160320606,160320601,160320600,160321326,160320613,160320614,160321016,160321017,160321194,160321179,160321082,160321167,
                        160321169,160321099,160321089,160321087,160321154,160321148,160321133,160274870,160274875,160274888,160274886,160274872,160274884,160274892,160275365,160275366,160274868,
                        160274874,160274793,160274781,160274780,160274905,160274893,160274894,160213616,160213676,160213677,160213666,160213668,160213553,160213555,160213617,160194008,160365818,160365822,
                        160177534,160177550
                        ]


    for edge_uid in additional_true:
        if final_iteration_edges_infra[final_iteration_edges_infra['edgeUID'] == edge_uid]['separated_bike_infrastructure_any'].item() == False:
            final_iteration_edges_infra.at[final_iteration_edges_infra[final_iteration_edges_infra['edgeUID'] == edge_uid].index.item(), 'separated_bike_infrastructure_any'] = True

    for edge_uid in additional_false:
        if final_iteration_edges_infra[final_iteration_edges_infra['edgeUID'] == edge_uid]['separated_bike_infrastructure_any'].item() == True:
            final_iteration_edges_infra.at[final_iteration_edges_infra[final_iteration_edges_infra['edgeUID'] == edge_uid].index.item(), 'separated_bike_infrastructure_any'] = False

    final_iteration_edges_infra.to_csv(f'{SAVE_PATH}/datasets/strava_edges_with_infra_final_iteration.csv', index=False)
    infra_gridv2.to_csv(f'{SAVE_PATH}/datasets/infrastructure_grid_final_run.csv')
        

    infra_gridv2['infrastructure'] = infra_gridv2['infrastructure'].astype(str).map({'False':False, 'True':True})

    for column in infra_gridv2.columns:
        if column not in ['infrastructure', 'geometry']:
            infra_gridv2[column] = pd.to_numeric(infra_gridv2[column], errors='coerce')

    fig_edge_infra_matching,axes = plt.subplot_mosaic([['.','.','1','1','1','1','.','.'],
                            ['s4','s5','s6','s7','s8','s9','s10','s11'],
                            ['s12','s13','s14','s15','s17','s18','s19','s20']],
                            figsize=(cm_to_inches(TOTAL_PAGE_WIDTH_CM), cm_to_inches(TOTAL_PAGE_WIDTH_CM/ 1.5)), dpi=500,
                            gridspec_kw={'height_ratios': [3, 1, 1]})


    infra_gridv2.plot(ax=axes['1'], column=f'{args.measure_to_consider}_treshold', cmap='viridis', alpha=0.5)

    infra_gridv2[infra_gridv2['infrastructure'] == True].plot(ax=axes['1'], facecolor='none', edgecolor='black', linewidth=0.25, label='Grid cells with Infrastructure')

    infra_gridv2[infra_gridv2['infrastructure'] == False].plot(ax=axes['1'], facecolor='white', edgecolor='black', linewidth=0.25, label='Grid cells') 
    axes['1'].set_xticks([])
    axes['1'].set_yticks([])
    infra_gridv2['iterative_column'] = infra_gridv2[f'{args.measure_to_consider}_treshold']
    for sigma in sigma_list:
        temp = infra_gridv2[['infrastructure',f"recall_pct_SIGMA{sigma}",f"f1_score_SIGMA{sigma}",f"precision_pct_SIGMA{sigma}",'geometry']].copy()
        temp.dropna(subset = [f'recall_pct_SIGMA{sigma}', f'f1_score_SIGMA{sigma}', f'precision_pct_SIGMA{sigma}'], inplace=True)
        temp.plot(ax = axes[f's{sigma}'],color = 'red')

        infra_gridv2.plot(ax=axes[f's{sigma}'], column=f'iterative_column', cmap='viridis', alpha=0.5)
        axes[f's{sigma}'].set_xticks([])
        axes[f's{sigma}'].set_yticks([])
        infra_gridv2.loc[temp.index, 'iterative_column'] = temp[f'f1_score_SIGMA{sigma}']

        infra_gridv2[infra_gridv2['infrastructure'] == True].plot(ax=axes[f's{sigma}'], facecolor='none', edgecolor='black', linewidth=0.10, label='Grid cells with Infrastructure')

        infra_gridv2[infra_gridv2['infrastructure'] == False].plot(ax=axes[f's{sigma}'], facecolor='white', edgecolor='black', linewidth=0.10, label='Grid cells') 

        axes[f's{sigma}'].text(0.01, 1.02, fr'$\sigma = {sigma}$', transform=axes[f's{sigma}'].transAxes, ha='left', va='bottom',fontsize = 7)

    fig_edge_infra_matching.savefig(f"{SAVE_PATH}/figures/main/figure_8_edge_infra_matching.pdf", dpi=500)
