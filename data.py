import pandas as pd
import torch
from torch.utils.data import Dataset

# can the data be stored in a more compact form and transformed when the batches are extracted?
# stage 1: store X in uint8 and Y in int16
# stage 2: pack the bytes in X
class SingleCsvDataset(Dataset):
    def __init__(self, file_path, num_features, device='cpu'):
        self.df = pd.read_csv(file_path, header=None, nrows=1024*10)
        self.num_features = num_features
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.iloc[idx, 0:self.num_features]
        y = self.df.iloc[idx, self.num_features:]

        X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y = torch.tensor(y.values, dtype=torch.float32).to(self.device)

        return X, y
