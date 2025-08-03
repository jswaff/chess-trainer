import gzip
import mmap
import os.path
import shutil

import numpy as np
import torch
import torch.multiprocessing

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from config import CFG
from quantize import quantize
from scipy.sparse import csr_matrix

class MMEpdDataSet(Dataset):
    def __init__(self, file_path, batch_size):
        print("initializing dataset")
        self.batch_size = batch_size

        self.offsets = [0]
        with open(file_path, "r+b") as f:
            self.f_mmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            self.f_mmap.madvise(mmap.MADV_SEQUENTIAL)
            for idx, _ in enumerate(iter(self.f_mmap.readline, b""), start=1):
                if idx % batch_size == 0:   # // 2
                    self.offsets.append(self.f_mmap.tell())  # where *next* line would start

        self.offsets.pop() # no partial batches
    def __len__(self):
        return len(self.offsets)

    # the item being returned is a mini-batch
    def __getitem__(self, idx):

        cached_batch_file = 'cache/' + str(idx // 1000) + '/' + str(idx) + '.pt.gz'

        if os.path.exists(cached_batch_file):
            (Xs, Xs2, ys,ys2) = load_batch(cached_batch_file)
        else:
            # read batch from disk in EPD format
            self.f_mmap.seek(self.offsets[idx])
            self.f_mmap.madvise(mmap.MADV_SEQUENTIAL)

            # encode batch
            indices = []
            indptr = [0]
            indices_f = [] # flipped
            indptr_f = [0]
            wscores = []
            win_ratios = []

            for _ in range(self.batch_size):
                line = self.f_mmap.readline()
                if line:
                    line_str = line.decode("utf-8")
                else:
                    break
                epd, score, wins, draws, losses = line_str.split(',')
                ind, ind_f, wscore, win_ratio = encode(epd, float(score), int(wins), int(draws), int(losses))

                indices += ind
                indptr += [indptr[-1] + len(ind)]

                indices_f += ind_f
                indptr_f += [indptr_f[-1] + len(ind_f)]

                wscores.append(wscore)
                win_ratios.append(win_ratio)

            # construct sparse feature tensors
            data = [1] * len(indices)
            coo = csr_matrix((data, indices, indptr), shape=(self.batch_size, 768), dtype=np.int8).tocoo()
            Xs = torch.sparse_coo_tensor(
                torch.LongTensor(np.vstack((coo.row, coo.col))),
                torch.FloatTensor(coo.data),
                torch.Size([self.batch_size, 768]))

            data = [1] * len(indices_f)
            coo = csr_matrix((data, indices_f, indptr_f), shape=(self.batch_size, 768), dtype=np.int8).tocoo()
            Xs2 = torch.sparse_coo_tensor(
                torch.LongTensor(np.vstack((coo.row, coo.col))),
                torch.FloatTensor(coo.data),
                torch.Size([self.batch_size, 768]))

            # construct label tensor
            ys = torch.from_numpy(np.array(wscores)).unsqueeze(1).to(torch.float32)
            ys2 = torch.from_numpy(np.array(win_ratios)).unsqueeze(1).to(torch.float32)

            # cache for later
            save_batch(cached_batch_file, Xs, Xs2, ys, ys2)

        return Xs, Xs2, ys, ys2

def custom_collate_fn(batch):
    # batch is a list, and should always be length=1 since batching is managed by the dataset
    assert(len(batch)==1)
    assert(len(batch[0])==4)

    Xs = batch[0][0]
    Xs2 = batch[0][1]
    ys = batch[0][2]
    ys2 = batch[0][3]

    return [Xs, Xs2, ys, ys2]

def load_batch(filename):
    with gzip.open(filename, 'rb') as f:
        data = torch.load(f)
    #Xs = data['Xs']
    #Xs2 = data['Xs2']
    Xs = torch.sparse_coo_tensor(data['Xs_i'], data['Xs_v'], data['Xs_shape'])
    Xs2 = torch.sparse_coo_tensor(data['Xs2_i'], data['Xs2_v'], data['Xs2_shape'])
    ys = data['ys']
    ys2 = data['ys2']
    return Xs,Xs2,ys,ys2

def save_batch(filename, Xs, Xs2, ys, ys2):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Xs = Xs.coalesce()
    Xs2 = Xs2.coalesce()

    data = {
        #'Xs': Xs,
        #'Xs2': Xs2,
        'Xs_i': Xs.indices(),
        'Xs_v': Xs.values(),
        'Xs_shape': Xs.shape,
        'Xs2_i': Xs2.indices(),
        'Xs2_v': Xs2.values(),
        'Xs2_shape': Xs2.shape,
        'ys': ys,
        'ys2': ys2
    }

    with gzip.open(filename, 'wb') as f:
        torch.save(data, f)

def encode(epd, score, wins, draws, losses):
    epd_parts = epd.split(" ")
    ranks = epd_parts[0].split("/")
    ptm = epd_parts[1]

    pieces = ['P','p','N','n','B','b','R','r','Q','q','K','k']

    ind = []
    flipped_ind = []
    sq = 0
    for rank_ind, rank in enumerate(ranks):
        for ch in rank:
            if '1' <= ch <= '8':
                sq += int(ch)
            elif ch in pieces:
                ind.append(pieces.index(ch) * 64 + sq)
                flipped_ind.append(pieces.index(ch.swapcase()) * 64 + (sq ^ 56))
                sq += 1
            else:
                raise Exception(f'invalid FEN character {ch} in {epd_parts[0]}')

    assert(sq==64)

    assert(2 <= len(ind) <= 32)
    assert(2 <= len(flipped_ind) <= 32)

    score = score / 100.0 # centi-pawns to pawns
    if ptm == 'w':
        wscore = score
    elif ptm == 'b':
        wscore = -score
    else:
        raise Exception(f'invalid ptm {ptm}')

    win_ratio = (wins + 0.5 * draws) / (wins + draws + losses)
    assert 0.0 <= win_ratio <= 1.0
    # normalize to [-1, 1]
    win_ratio = win_ratio * 2.0 - 1.0

    return ind, flipped_ind, wscore, win_ratio

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

    # note: batching is handled by the dataset, not the dataloader
    train_dl = DataLoader(dataset=dataset, sampler=train_sampler, collate_fn=custom_collate_fn, num_workers=CFG.num_workers, pin_memory=True)
    test_dl = DataLoader(dataset=dataset, sampler=test_sampler, collate_fn=custom_collate_fn, num_workers=CFG.num_workers, pin_memory=True)
    valid_dl = DataLoader(dataset=dataset, sampler=valid_sampler, collate_fn=custom_collate_fn, num_workers=CFG.num_workers, pin_memory=True)

    return train_dl, test_dl, valid_dl

def torch_get_weights(tensor_row):
    return "\n".join([str(tensor.item()) for tensor in tensor_row]) + "\n"

def save_model(model, filename, use_quantization=False):
    with open(filename, "w") as file:
        for parameter in model.parameters():
            if parameter.dim() == 1:
                row = quantize(parameter.data, 64) if use_quantization else parameter.data
                file.write(torch_get_weights(row))
            elif parameter.dim() == 2:
                params = quantize(parameter.data, 64) if use_quantization else parameter.data
                for row in params:
                    file.write(torch_get_weights(row))
            else:
                assert 0

def clear_cache_dir():
    for filename in os.listdir('cache'):
        file_path = os.path.join('cache', filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
