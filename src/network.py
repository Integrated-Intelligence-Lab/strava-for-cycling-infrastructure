import torch
import torch.nn as nn
import numpy as np

class MultiHeadNetwork(nn.Module):
    def __init__(self, 
                 input_dim = 8, 
                 num_shared_layers=2, 
                 shared_width=128, 
                 num_head_layers=2, 
                 head_width=32, 
                 num_heads=18,
                 activation=nn.SiLU()):
        """
        Parameters:
          input_dim:         Dimensionality of the input features.
          num_shared_layers: Number of fully connected layers in the shared module.
          shared_width:      Number of neurons in each shared layer.
          num_head_layers:   Number of layers in each head.
                             If set to 1, the head is a single linear layer mapping from shared_width to 1.
                             If >1, the head includes hidden layers with the given head_width.
          head_width:        Number of neurons in each hidden head layer (used if num_head_layers > 1).
          num_heads:         Number of heads (i.e. distinct outputs).
          activation:        Activation function to use (default is ReLU).
        """
        super(MultiHeadNetwork, self).__init__()
        
        # Build shared layers
        shared_layers = []
        for i in range(num_shared_layers):
            # First layer goes from input_dim to shared_width, subsequent layers are shared_width->shared_width
            in_features = input_dim if i == 0 else shared_width
            shared_layers.append(nn.Linear(in_features, shared_width))
            shared_layers.append(activation)
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Build separate heads for each counter
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head_layers = []
            # If there's only one head layer, map directly from shared representation to output.
            if num_head_layers == 1:
                head_layers.append(nn.Linear(shared_width, 1))
            else:
                # First head layer: from shared representation to head_width
                head_layers.append(nn.Linear(shared_width, head_width))
                head_layers.append(activation)
                # If additional head layers are requested, add hidden layers.
                for _ in range(num_head_layers - 2):
                    head_layers.append(nn.Linear(head_width, head_width))
                    head_layers.append(activation)
                # Final head layer: from head_width to 1 output
                head_layers.append(nn.Linear(head_width, 1))
            self.heads.append(nn.Sequential(*head_layers))
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        Returns:
          Tensor of shape (batch_size, num_heads) with one prediction per head.
        """
        # Compute the shared representation
        shared_rep = self.shared_layers(x)
        # Get predictions from each head
        outputs = [head(shared_rep) for head in self.heads]  # Each head outputs (batch_size, 1)
        # Concatenate the outputs from all heads along dimension 1
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
def train_loop(dataloader, model, loss_fn, optimizer,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()
    total_loss = 0
    for batch, (X, y,id) in enumerate(dataloader):
        X,y,id = X.to(device),y.to(device),id.to(device)
        indices = id.to(torch.int64)
        indices = indices - 1
        outputs = model(X)
        pred = outputs.gather(dim=1, index=indices.unsqueeze(1)).squeeze(1)  
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.cpu().detach().numpy()    

    

    return total_loss

def test_loop(dataloader, model,
              max_min_per_counter,
              device='cuda' if torch.cuda.is_available() else 'cpu'
              ):

    model.eval()
    preds_o, targets_o = [],[]
    with torch.no_grad():
        for (X, y,id) in dataloader:

            counter_indices = id.to(device) -1  
            counter_indices =  counter_indices.to(torch.int64)
            X_input = X.to(device)
            y = y.to(device)
            y = y.squeeze(1)
            print(y.shape)

            pred = model(X_input)
            pred_selected = pred.gather(dim=1, index=counter_indices.unsqueeze(1)).squeeze(1)


            max_values = torch.tensor([max_min_per_counter[str((counter_idx + 1).item())]['max']
                                    for counter_idx in counter_indices],
                                    dtype=pred_selected.dtype, device=pred_selected.device)
            min_values = torch.tensor([max_min_per_counter[str((counter_idx + 1).item())]['min']
                                    for counter_idx in counter_indices],
                                    dtype=pred_selected.dtype, device=pred_selected.device)


            pred_unnormalised = pred_selected * (max_values - min_values) + min_values
            targets_unnormalised = y * (max_values - min_values) + min_values
            print(targets_unnormalised.shape)
            preds_o.append(pred_unnormalised.cpu().numpy())
            targets_o.append(targets_unnormalised.cpu().numpy())
 
        
        preds_o = np.array(preds_o)
        targets_o = np.array(targets_o)
        preds_o = np.concatenate(preds_o).ravel()      
        targets_o = np.concatenate(targets_o).ravel()     
        MAE = np.abs(preds_o - targets_o)
        MAPE = np.abs((preds_o - targets_o) / targets_o) * 100

        ss_res = np.sum((preds_o- targets_o) ** 2)
        ss_tot = np.sum((targets_o - targets_o.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot

    return np.mean(MAE),np.mean(MAPE),r2