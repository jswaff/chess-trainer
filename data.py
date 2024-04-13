import mmap

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from config import CFG
from epd import to_one_hot


class EpdDataset(Dataset):
    def __init__(self, file_path, num_samples=None, device='cpu'):
        self.df = pd.read_csv(file_path, header=None, nrows=num_samples)
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = self.df.iloc[idx, 0]
        y = y[..., np.newaxis]
        epd = self.df.iloc[idx, 1]
        X, ptm = to_one_hot(epd)
        if ptm == 'b':
            y = -y

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        return X, y


class MMEpdDataSet(Dataset):
    def __init__(self, file_path, device='cpu'):
        self.device = device

        self.offsets = [0]
        with open(file_path, "r+b") as f:
            self.f_mmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            for _ in iter(self.f_mmap.readline, b""):
                self.offsets.append(self.f_mmap.tell())  # where *next* line would start
            self.offsets.pop()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.f_mmap.seek(self.offsets[idx])
        line_bytes = self.f_mmap.readline()
        line_str = str(line_bytes, encoding='utf-8').rstrip()
        y, epd = line_str.split(',', 1)
        y = int(y)
        X, ptm = to_one_hot(epd)
        if ptm == 'b':
            y = -y

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor([y], dtype=torch.float32).to(self.device)

        return X, y


def build_data_loaders():
    #dataset = EpdDataset(file_path=CFG.input_path, num_samples=CFG.num_samples, device=CFG.device)
    dataset = MMEpdDataSet(file_path=CFG.input_path, device=CFG.device)
    dataset_size = len(dataset)
    print(f'dataset_size: {dataset_size}')
    indices = list(range(dataset_size))
    test_dataset_size = int(np.floor(0.1 * dataset_size))
    valid_dataset_size = test_dataset_size

    np.random.seed()
    np.random.shuffle(indices)
    train_indices, test_indices = indices[test_dataset_size:], indices[:test_dataset_size]
    train_indices, valid_indices = train_indices[valid_dataset_size:], train_indices[:valid_dataset_size]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=CFG.batch_size, sampler=train_sampler,
                          num_workers=CFG.num_workers)
    test_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=CFG.batch_size, sampler=test_sampler,
                         num_workers=CFG.num_workers)
    valid_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=CFG.batch_size, sampler=valid_sampler,
                          num_workers=CFG.num_workers)

    return train_dl, len(train_indices), test_dl, len(test_indices), valid_dl, len(valid_indices)
