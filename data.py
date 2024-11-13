import gzip
import mmap
import os.path
import pickle

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.multiprocessing
from config import CFG


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

        encoded_batch_file = 'data/' + str(idx // 1000) + '/' + str(idx) + '.pickle'

        if os.path.exists(encoded_batch_file):
            (Xs_tensor, Xs2_tensor, ys_tensor) = load_encoded_batch(encoded_batch_file)
        else:
            self.f_mmap.seek(self.offsets[idx])
            self.f_mmap.madvise(mmap.MADV_SEQUENTIAL)

            Xs = np.zeros(shape=(self.batch_size, 768), dtype=np.float32)
            Xs2 = np.zeros(shape=(self.batch_size, 768), dtype=np.float32)
            ys = np.zeros(shape=(self.batch_size, 1), dtype=np.float32)

            lines_read = 0
            for i in range(self.batch_size):
                line = self.f_mmap.readline()
                if line:
                    line_str = line.decode("utf-8")
                    lines_read = lines_read + 1
                else:
                    break
                y, epd = line_str.split(',', 1)
                encode(epd, int(y), Xs, Xs2, ys, i)

            Xs_tensor = torch.tensor(Xs[:lines_read, :], dtype=torch.float32)
            Xs2_tensor = torch.tensor(Xs2[:lines_read, :], dtype=torch.float32)
            ys_tensor = torch.tensor(ys[:lines_read, :], dtype=torch.float32)

            # cache for later
            save_encoded_batch(encoded_batch_file, Xs_tensor, Xs2_tensor, ys_tensor)

        return Xs_tensor, Xs2_tensor, ys_tensor

def load_encoded_batch(fname):
    with gzip.GzipFile(fname, 'rb') as file:
        (Xs,Xs2,ys) = pickle.load(file)
        return Xs,Xs2,ys

def save_encoded_batch(fname, Xs, Xs2, ys):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with gzip.GzipFile(fname, 'wb') as file:
        pickle.dump((Xs,Xs2,ys), file)

# Xs : one hot representation of position in epd
# Xs2 : one hot representation of the position in epd flipped (a white rook on A1 is now a black rook on A8)
# ys : score (label) of epd, from white's perspective
def encode(epd, score, Xs, Xs2, ys, idx):
    epd_parts = epd.split(" ")
    ranks = epd_parts[0].split("/")
    ptm = epd_parts[1]

    offsets = {
        'R' :   0,
        'r' :  64,
        'N' : 128,
        'n' : 192,
        'B' : 256,
        'b' : 320,
        'Q' : 384,
        'q' : 448,
        'K' : 512,
        'k' : 576,
        'P' : 640,
        'p' : 704
    }

    sq = 0
    for rank_ind, rank in enumerate(ranks):
        col_ind = 0
        for ch in rank:
            if '1' <= ch <= '8':
                col_ind += int(ch)
                sq += int(ch)
            elif ch in offsets.keys():
                Xs[idx][offsets[ch] + sq] = 1
                flipped_sq = abs(7-rank_ind)*8 + col_ind
                Xs2[idx][offsets[ch.swapcase()] + flipped_sq] = 1
                col_ind += 1
                sq += 1
            else:
                raise Exception(f'invalid FEN character {ch}')
    if sq != 64:
        raise Exception(f'invalid square count {sq}')

    # label is score from white's perspective
    if ptm == 'w':
        ys[idx][0] = score
    elif ptm == 'b':
        ys[idx][0] = -score
    else:
        raise Exception(f'invalid ptm {ptm}')


def build_data_loaders():
    print(f'data_path: {CFG.data_path}')
    dataset = MMEpdDataSet(file_path=CFG.data_path, batch_size=CFG.batch_size)
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

    train_dl = DataLoader(dataset=dataset, sampler=train_sampler, num_workers=CFG.num_workers, pin_memory=True)
    test_dl = DataLoader(dataset=dataset, sampler=test_sampler, num_workers=CFG.num_workers, pin_memory=True)
    valid_dl = DataLoader(dataset=dataset, sampler=valid_sampler, num_workers=CFG.num_workers, pin_memory=True)

    return train_dl, test_dl, valid_dl
