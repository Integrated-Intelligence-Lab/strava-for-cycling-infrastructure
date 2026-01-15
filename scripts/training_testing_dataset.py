import pandas as pd

from dotenv import load_dotenv
import os

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
import pickle
from tqdm import tqdm
import numpy as np

from src.network_datasets import Windowed
import torch

    

def create_test_val_test_indices(dataset,seq_len = 7,id_col = "identifier_counter_int",
                                 train_ratio = 0.85, val_ratio = 0.05, test_ratio = 0.1
                                 ):
    train_idx, val_idx, test_idx = [], [], []

    length_train_idx = {}
    for cid in sorted(dataset[id_col].unique()):
        df = dataset[dataset[id_col] == cid]
        
        idx_c = dataset.index[dataset[id_col] == cid].to_numpy()   # raw DF indices for this counter

        if len(idx_c) <= seq_len:
            continue

        # map DF indices ≥ seq_len into dataset indices:
        ds_idx_c = idx_c[seq_len:] - seq_len             # Guarantees 30 for the test set?
        N = len(ds_idx_c)

        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        t_end = int(train_ratio * N)
        v_end = int((train_ratio + val_ratio) * N)

        train_idx.extend(ds_idx_c[:t_end ])
        val_idx.extend(  ds_idx_c[t_end:v_end])
        test_idx.extend( ds_idx_c[v_end:])
        length_train_idx[cid] = len(ds_idx_c[:t_end])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx), length_train_idx

def create_dataframes_for_fcnn(X,indices,
                               seq_len,y_norm, ids,
                               dataset_split = 'train'):
    ds_fcnn = pd.DataFrame(columns=["X","y_pred",'id'])
    ds  = Windowed(X, id_idx[indices], y_norm[indices], seq_len)

    for row in tqdm(ds, desc=f"Creating FCNN {dataset_split} dataset"):
        x_seq,_, y_next, id_next = row
        
        x_feature_total = np.concatenate((x_seq.numpy()[:-1,0], x_seq[-1].numpy()))
        ds_fcnn.loc[len(ds_fcnn)] = {"X":x_feature_total,"y_pred": y_next.numpy(),'id':id_next.numpy()+1}
        
    return ds_fcnn

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--normalization', type=str, default='minmax', help='Normalization method to use: minmax or log')

    args = parser.parse_args()
    """
    Loading in the total merged dataset and creating train, validation datasets
    1. We remove the years that are known to have issues for specific counters
    2. We make the counter id start from 0
    3. We enforce dtypes and chronological order within every counter
    """
    
    with open(f'{DATA_PATH}/bike_counters/counter_years_to_exclude.pkl', 'rb') as f:
        counter_years_to_exclude = pickle.load(f)
    
    total_merged_dataset = pd.read_csv(f"{DATA_PATH}/total_merged_dataset.csv")

    # We remove from the dataset the years that are known to have issues for specific counters (The correlation is < 0)
    print("Initial dataset size:", total_merged_dataset.shape)
    for device_name in counter_years_to_exclude:
        for year in counter_years_to_exclude[device_name]:
            total_merged_dataset = total_merged_dataset.drop(total_merged_dataset[(total_merged_dataset['identifier_counter'] == device_name) & (total_merged_dataset['year'] == year)].index)
    print("Cleaned dataset size:", total_merged_dataset.shape)
    # We make the counter id start from 0
    total_merged_dataset['identifier_counter_int'] = total_merged_dataset['identifier_counter_int'] -1

    id_col     = "identifier_counter_int"
    id_idx  = total_merged_dataset["identifier_counter_int"].values         
    date_col   = "date"

    # Enforce dtype and chronological order within every counter
    total_merged_dataset[id_col] = total_merged_dataset[id_col].astype(np.int32)

    total_merged_dataset[date_col] = pd.to_datetime(total_merged_dataset[date_col])
    total_merged_dataset = total_merged_dataset.sort_values([id_col, date_col])

    total_merged_dataset = total_merged_dataset.reset_index(drop=True)

    if args.normalization == 'log':
        target_col  = ['counter_trips_log']
        strava_col   = ['strava_trips_log']
    elif args.normalization == 'minmax':
        target_col = ["counter_trips_normalised"]
        strava_col = ['strava_trips_normalised']
    else:
        raise ValueError("Normalization method not recognized. Use 'minmax' or 'log'.")

    cont_cols   = ['temp_avg','precip_quantity','wind_speed_10m']
    cyclic_cols = ["month_sin", "month_cos", "wd_sin", "wd_cos","year_normalised", "is_holiday","any_rain"]   
        
    if args.normalization == 'log':
        preprocessor = ColumnTransformer([
            ("scale_strava", StandardScaler(), strava_col),
            ("pass",  "passthrough",  cyclic_cols),
            ("scale_rest", StandardScaler(), cont_cols),
            
        ])
    else:
        preprocessor = ColumnTransformer([
            ("pass",  "passthrough", strava_col + cyclic_cols),
            ("scale_rest", StandardScaler(), cont_cols),
        ])

    train_idx, val_idx, test_idx, length_train_idx = create_test_val_test_indices(total_merged_dataset,seq_len=7,id_col=id_col)

    y_norm = total_merged_dataset[target_col].astype(np.float32).values  # (N,)
    X_train   =   preprocessor.fit_transform(total_merged_dataset.iloc[train_idx]).astype(np.float32)
    X_val     =   preprocessor.transform(total_merged_dataset.iloc[val_idx]).astype(np.float32)
    X_test    =   preprocessor.transform(total_merged_dataset.iloc[test_idx]).astype(np.float32)

    ds_fcnn_train = create_dataframes_for_fcnn(X_train,train_idx,seq_len=7,y_norm=y_norm,ids=id_idx,dataset_split='train')
    ds_fcnn_val   = create_dataframes_for_fcnn(X_val,val_idx,seq_len=7,y_norm=y_norm,ids=id_idx,dataset_split='validation')
    ds_fcnn_test  = create_dataframes_for_fcnn(X_test,test_idx,seq_len=7,y_norm=y_norm,ids=id_idx,dataset_split='test')

    torch.save(ds_fcnn_train, f"{DATA_PATH}/network/ds_fcnn_train_normalization_{args.normalization}.pt")
    torch.save(ds_fcnn_val, f"{DATA_PATH}/network/ds_fcnn_val_normalization_{args.normalization}.pt")
    torch.save(ds_fcnn_test, f"{DATA_PATH}/network/ds_fcnn_test_normalization_{args.normalization}.pt")