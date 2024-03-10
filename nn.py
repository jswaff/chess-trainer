import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from data import EpdDataset

num_features = 768
batch_size = 256
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("device: ", device)

dataset = EpdDataset(file_path='data/mat0-fen.csv', device=device)
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

train_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
test_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, num_workers=8)
valid_dl = DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=8)

# define the model
model = nn.Sequential(
    nn.Linear(num_features, 1)
    , nn.Identity()
).to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)  # try weight decay?
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    min_loss = np.inf
    training_start_time = time.time()
    no_improvement_cnt = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
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

        # update best
        improvement=0
        if loss_hist_valid[epoch] < min_loss:
            improvement = min_loss - loss_hist_valid[epoch]
            min_loss = loss_hist_valid[epoch]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss,
                'loss_hist_train': loss_hist_train,
                'loss_hist_valid': loss_hist_valid,
            }, 'model-out.pt')

        if improvement < 0.0001:
            no_improvement_cnt = no_improvement_cnt+1

        print(f'Epoch {epoch + 1} loss: {loss_hist_valid[epoch]:.4f} ',
              'epoch time: {:.2f}m'.format((time.time()-epoch_start_time) / 60),
              'total time: {:.2f}m'.format((time.time()-training_start_time) / 60),
              f'improvement: {improvement:.4f} no_improvement_cnt: {no_improvement_cnt}')

        if no_improvement_cnt > 5:
            print('Early exit triggered.')
            loss_hist_train = loss_hist_train[0:epoch]
            loss_hist_valid = loss_hist_valid[0:epoch]
            break

    print('Total training time: {:.2f}m'.format((time.time()-training_start_time) / 60))
    return min_loss, best_weights, loss_hist_train, loss_hist_valid


# train
torch.manual_seed(1)
num_epochs = 2
hist = train(model, num_epochs, train_dl, valid_dl)
print(f'Min validation loss: {hist[0]}')
print(f'Best validation weights: {hist[1]}')

# measure performance against test set
loss_test = 0
for x_batch, y_batch in test_dl:
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss_test += loss.item()
loss_test /= len(test_dl.dataset)
print(f'Test loss: {loss_test}')

# visualize learning curves
#import matplotlib.pyplot as plt
#x_arr = np.arange(len(hist[2])) + 1
#fig = plt.figure(figsize=(12,4))
#ax = fig.add_subplot(1,2,1)
#ax.plot(x_arr, hist[2], '-o', label='Train loss')
#ax.plot(x_arr, hist[3], '--<', label='Validation loss')
#ax.legend(fontsize=15)
#plt.show()

