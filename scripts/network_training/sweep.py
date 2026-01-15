from src.network import MultiHeadNetwork, train_loop_multihead, test_loop_multihead
from src.network_datasets import FCNNDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wandb

from dotenv import load_dotenv
import os
import json
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
SAVE_PATH = os.getenv("RESULTS_PATH")
TOTAL_PAGE_WIDTH_CM = float(os.getenv("FULL_PAGE_WIDTH_CM"))
TOTAL_PAGE_LENGTH_CM = float(os.getenv("FULL_PAGE_LENGTH_CM"))

ds_tr_fcnn = torch.load(f"{DATA_PATH}/network/Datasets_processed/ds_fcnn_train_normalization_minmax.pt")
ds_val_fcnn = torch.load(f"{DATA_PATH}/network/Datasets_processed/ds_fcnn_val_normalization_minmax.pt")
ds_test_fcnn = torch.load(f"{DATA_PATH}/network/Datasets_processed/ds_fcnn_test_normalization_minmax.pt")

ds_custom_tr = FCNNDataset(ds_tr_fcnn['X'], ds_tr_fcnn['y_pred'], ds_tr_fcnn['id'])
ds_custom_val = FCNNDataset(ds_val_fcnn['X'], ds_val_fcnn['y_pred'],ds_val_fcnn['id'])
ds_custom_te = FCNNDataset(ds_test_fcnn['X'], ds_test_fcnn['y_pred'],ds_test_fcnn['id'])

dl_opts  = dict(pin_memory=device.type == "cuda",
                    drop_last=True)

dl_tr_fcnn  = DataLoader(ds_custom_tr,batch_size=10000 ,shuffle=True)
dl_val_fcnn = DataLoader(ds_custom_val,batch_size=400, shuffle=False, **dl_opts)
dl_te_fcnn  = DataLoader(ds_custom_te,batch_size=200, shuffle=False, **dl_opts)


loss_fn_fcnn = nn.MSELoss()

best_test_mae = 1000000
best_test_mae_run_id = ''


PATIENCE = 25

with open(f"{SAVE_PATH}/datasets/sweep_results/FCNN_best_test_mae_{os.environ['SLURM_JOB_ID']}.json", "w") as f:
       json.dump({
        'best_test_mae': best_test_mae,
        'best_test_mae_run_id': best_test_mae_run_id,
}, f)
       

def objective_sweep_multihead(config):
    losses_list = []

    multi_head_model = MultiHeadNetwork(input_dim=11,
                 num_shared_layers=config.num_shared_layers,
                 shared_width=config.shared_width, 
                 num_head_layers=config.num_head_layers, 
                 head_width=config.head_width, 
                 num_heads=18).to(device)
    
    number_of_parameters = sum(p.numel() for p in multi_head_model.parameters())
    optimizer = torch.optim.AdamW(multi_head_model.parameters(), lr=config.learning_rate)

    with open(f"{SAVE_PATH}/datasets/sweep_results/FCNN_best_test_mae_{os.environ['SLURM_JOB_ID']}.json", "r") as f:
        best_test_mae_data = json.load(f)

    best_test_mae_previous_runs = best_test_mae_data["best_test_mae"]
    best_test_mae_run_id = best_test_mae_data['best_test_mae_run_id']

    
    print("==========================================")
    print(f"Current best test_mae: {best_test_mae_previous_runs}")
    print(f"Current best_test_mae_run_id: {best_test_mae_run_id}")
    best_test_mae = best_test_mae_previous_runs
    best_val, epochs_no_imp = float("inf"), 0

    for t in range(config.epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop_multihead(dl_tr_fcnn, multi_head_model, loss_fn_fcnn, optimizer)
        losses_list.append(loss)

        ## ---- validation ----
        val_mae,_,_ = test_loop_multihead(dl_val_fcnn, multi_head_model)

        if val_mae + 1e-4 < best_val:
            best_val = val_mae
            epochs_no_imp = 0
            
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                print("Early stopping.")
                break


    ## ---- testing ----
    test_mae,test_mape,r2 = test_loop_multihead(dl_te_fcnn, multi_head_model)
    
    if test_mae < best_test_mae:
        torch.save(multi_head_model.state_dict(), f"{SAVE_PATH}/datasets/sweep_results/FCNN_best_model_weights_sweep_{os.environ['SLURM_JOB_ID']}.pth")
        print(f"Model saved with test mae: {test_mae}")
        best_test_mae = test_mae

    if best_test_mae < best_test_mae_previous_runs:
        best_test_mae_run_id = wandb.run.id
        with open(f"{SAVE_PATH}/datasets/sweep_results/FCNN_best_test_mae_{os.environ['SLURM_JOB_ID']}.json", "w") as f:
            json.dump({
                'best_test_mae': best_test_mae.item(),
                'best_test_mae_run_id': best_test_mae_run_id,
            }, f)

    return test_mae,test_mape,r2,number_of_parameters 

FCNN_sweep_configuration = {
        "name": "FCNN Multihead network",
        "method": "random",
        "metric": {"goal": "minimize", "name": "Final MAE"},
        "parameters": {
            "learning_rate": {"min": 0.00001,"max": 0.005},
            "num_shared_layers": {"min":1,"max": 4},
            "shared_width": {"min": 8,"max": 256},
            "num_head_layers": {"min": 1,"max": 4},
            "head_width": {"min": 4,"max": 64},
            "dropout":{"min":0.0001,"max": 0.3},
            "epochs":{"min": 100,"max": 500}
        }
}
def main():
    wandb.init(dir = "/wandb")
    test_mae,test_mape,r2,number_of_parameters  = objective_sweep_multihead(wandb.config)

    wandb.log({"Final MAE": test_mae,"Final MAPE":test_mape,"Final R2":r2,'Number of parameters':number_of_parameters,'job_id' :os.environ['SLURM_JOB_ID']})

if __name__ == "__main__":
    os.environ['SLURM_JOB_ID'] = "0"  # for testing purposes

    sweep_id = wandb.sweep(sweep=FCNN_sweep_configuration, project="strava_for_cycling_infrastructure")
    wandb.agent(sweep_id, function=main,count = 1)