import numpy as np
import pandas as pd
import torch
import EPD
from torch.utils.data import Dataset


class EpdDataset(Dataset):
    def __init__(self, file_path, device='cpu'):
        self.df = pd.read_csv(file_path, header=None)
        self.device = device
        self.epd_parser = EPD.EPD()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = self.df.iloc[idx, 0]
        y = y[..., np.newaxis]
        epd = self.df.iloc[idx, 1]
        X,ptm = self.epd_parser.toOHE(epd)
        if ptm=='b':
            y = -y

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        return X,y

#dataset = EpdDataset(file_path='data/mat0-fen.csv')
#print(f'len {dataset.__len__()}')
#dataset.__getitem__(2)


# model = nn.Sequential(
#     nn.Linear(768, 1)
#     , nn.Identity()
# ).to("cpu")
# model.load_state_dict(torch.load("model-out.pt"))
#print(torch.load("model-out.pt"))


