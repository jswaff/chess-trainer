import copy
import time
import numpy as np
import torch

from config import CFG


# note: the dataset used in the data loaders is the full dataset, so we can't calculate
# loss using len(dl.dataset)
def train(model, num_epochs, train_dl, train_sz, valid_dl, valid_sz, loss_fn, optimizer):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    min_loss = np.inf
    best_weights = copy.deepcopy(model.state_dict())
    training_start_time = time.time()
    early_stop_cnt = 5
    print('Epoch        Train        Valid        ETime        TTime        Delta        ESCnt')
    print('-----------------------------------------------------------------------------------')
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
        loss_hist_train[epoch] /= train_sz

        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
            loss_hist_valid[epoch] /= valid_sz

        # update best
        delta = 0
        if loss_hist_valid[epoch] < min_loss:
            delta = loss_hist_valid[epoch] - min_loss
            min_loss = loss_hist_valid[epoch]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss,
                'loss_hist_train': loss_hist_train,
                'loss_hist_valid': loss_hist_valid,
            }, CFG.model_name)

        if delta >= 0:
            early_stop_cnt = early_stop_cnt - 1

        print(f'{epoch + 1:>5}',
              f'{loss_hist_train[epoch]:>12.4f}',
              f'{loss_hist_valid[epoch]:>12.4f}',
              '{:12.2f}'.format((time.time() - epoch_start_time) / 60),
              '{:12.2f}'.format((time.time() - training_start_time) / 60),
              f'{delta:>12.4f}',
              f'{early_stop_cnt:>12}')

        if early_stop_cnt == 0:
            print('Early exit triggered.')
            loss_hist_train = loss_hist_train[0:epoch]
            loss_hist_valid = loss_hist_valid[0:epoch]
            break

    print('Total training time: {:.2f}m'.format((time.time() - training_start_time) / 60))
    return min_loss, best_weights, loss_hist_train, loss_hist_valid
