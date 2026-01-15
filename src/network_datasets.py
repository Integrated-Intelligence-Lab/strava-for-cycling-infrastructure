import torch
from torch.utils.data import Dataset

class Windowed(Dataset):
    def __init__(self, X, ids, y, L):
        self.X, self.ids, self.y, self.L = (
            torch.tensor(X), torch.tensor(ids), torch.tensor(y), L
        )

    def __len__(self):
        return len(self.X) - self.L # we subtract L!

    def __getitem__(self, idx):
        x_seq  = self.X[idx : idx + self.L+1]        # (L, F_num)
        id_seq = self.ids[idx : idx + self.L+1]      # (L,)
        y_next = self.y[idx + self.L]              # scalar
        id_next = self.ids[idx + self.L]           # scalar
        return x_seq, id_seq, y_next, id_next
    
class FCNNDataset(Dataset):
    def __init__(self, X,y,id):
        self.train_vector = X
        self.target = y
        self.id = id

    def __len__(self):
        return len(self.train_vector)

    def __getitem__(self, idx):
        sample = self.train_vector[idx]
        label = self.target[idx]
        id = self.id[idx]
        
        return sample, label,id