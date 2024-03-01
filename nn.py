import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# the dataset class
# TODO: support for very large CSV files
class SingleCsvDataset(Dataset):
    def __init__(self, file_path, device='cpu'):
        self.df = pd.read_csv(file_path, header=None, nrows=1024*10)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.iloc[idx, 0:10]
        y = self.df.iloc[idx, 10:]

        X = torch.tensor(X.values, dtype=torch.float32).to(device)
        y = torch.tensor(y.values, dtype=torch.float32).to(device)

        return X, y

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device: ", device)

# data loader
dataset = SingleCsvDataset(file_path='data/mat0.csv', device=device)
print(dataset.__getitem__(0))
print(dataset.__getitem__(1))
print(dataset.__getitem__(2))

# create data indices for training and test splits
dataset_size = len(dataset)
print(f'dataset_size: {dataset_size}')
indices = list(range(dataset_size))
test_dataset_size = int(np.floor(0.1 * dataset_size))
valid_dataset_size = test_dataset_size
print(f'test_dataset_size: {test_dataset_size}')
print(f'valid_dataset_size: {valid_dataset_size}')
np.random.seed(42)
np.random.shuffle(indices)
train_indices, test_indices = indices[test_dataset_size:], indices[:test_dataset_size]
train_indices, valid_indices = train_indices[valid_dataset_size:], train_indices[:valid_dataset_size]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

batch_size = 256
train_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
test_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler)
valid_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_sampler)


# define the model
model = nn.Sequential(
    nn.Linear(10, 1)
    , nn.Identity()
).to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # TODO: optimal lr?  try Adam?
# optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    best_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
        loss_hist_train[epoch] /= len(train_dl.dataset)

        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            if loss_hist_valid[epoch] < best_loss:
                best_loss = loss_hist_valid[epoch]
                best_weights = copy.deepcopy(model.state_dict())
                # TOOD: save weights, reload option?

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1} loss: {loss_hist_valid[epoch]:.4f}')
            print(f'weights: {model.state_dict()}')

        # TODO: early stop
    return best_loss, best_weights, loss_hist_train, loss_hist_valid


# train
torch.manual_seed(1)
num_epochs = 300
hist = train(model, num_epochs, train_dl, valid_dl)
print(f'best validation loss: {hist[0]}')
print(f'best validation weights: {hist[1]}')

loss_test = 0
for x_batch, y_batch in test_dl:
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss_test += loss.item()
loss_test /= len(test_dl.dataset)
print(f'test loss: {loss_test}')

# visualize learning curves
import matplotlib.pyplot as plt
x_arr = np.arange(len(hist[2])) + 1
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr, hist[2], '-o', label='Train loss')
ax.plot(x_arr, hist[3], '--<', label='Validation loss')
ax.legend(fontsize=15)
plt.show()

