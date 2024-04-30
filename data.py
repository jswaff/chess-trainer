import mmap

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.multiprocessing
from config import CFG
from epd import to_one_hot


class MMEpdDataSet(Dataset):
    def __init__(self, file_path, batch_size):
        print("initializing dataset")
        self.batch_size = batch_size

        self.offsets = [0]
        with open(file_path, "r+b") as f:
            self.f_mmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            self.f_mmap.madvise(mmap.MADV_SEQUENTIAL)
            for idx, _ in enumerate(iter(self.f_mmap.readline, b""), start=1):
                if idx % batch_size == 0:
                    self.offsets.append(self.f_mmap.tell())  # where *next* line would start

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.f_mmap.seek(self.offsets[idx])
        self.f_mmap.madvise(mmap.MADV_SEQUENTIAL)

        Xs = np.ndarray(shape=(self.batch_size, 768))
        ys = np.ndarray(shape=(self.batch_size, 1))

        lines_read = 0
        for i in range(self.batch_size):
            line = self.f_mmap.readline()
            if line:
                line_str = line.decode("utf-8")
                lines_read = lines_read + 1
            else:
                break
            y, epd = line_str.split(',', 1)
            y = int(y)
            X, ptm = to_one_hot(epd)  # TODO: faster to set tensor directly?
            for j in range(768):
                Xs[i][j] = X[j]
            if ptm == 'b':
                y = -y
            ys[i][0] = y

        Xs = torch.tensor(Xs[:lines_read, :], dtype=torch.float32)
        ys = torch.tensor(ys[:lines_read, :], dtype=torch.float32)
        # Xs = torch.rand([512, 768], dtype=torch.float32)
        # ys = torch.rand([512, 1], dtype=torch.float32)

        return Xs, ys


def build_data_loaders():
    print(f'input_path: {CFG.input_path}')
    dataset = MMEpdDataSet(file_path=CFG.input_path, batch_size=CFG.batch_size)
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

    # when using the MMEpdDataSet, set batch_size=1 as it is already batching
    train_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=train_sampler,
                          num_workers=CFG.num_workers, pin_memory=True)
    test_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=test_sampler,
                         num_workers=CFG.num_workers, pin_memory=True)
    valid_dl = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=valid_sampler,
                          num_workers=CFG.num_workers, pin_memory=True)

    return train_dl, len(train_indices), test_dl, len(test_indices), valid_dl, len(valid_indices)
