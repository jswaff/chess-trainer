import copy
import time
import numpy as np
import torch

from config import CFG


def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    min_loss = np.inf
    best_weights = copy.deepcopy(model.state_dict())
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
        improvement = 0
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
            }, CFG.model_name)

        if improvement < 0.0001:
            no_improvement_cnt = no_improvement_cnt + 1

        print(f'Epoch {epoch + 1} loss: {loss_hist_valid[epoch]:.4f} ',
              'epoch time: {:.2f}m'.format((time.time() - epoch_start_time) / 60),
              'total time: {:.2f}m'.format((time.time() - training_start_time) / 60),
              f'improvement: {improvement:.4f} no_improvement_cnt: {no_improvement_cnt}')

        if no_improvement_cnt > 5:
            print('Early exit triggered.')
            loss_hist_train = loss_hist_train[0:epoch]
            loss_hist_valid = loss_hist_valid[0:epoch]
            break

    print('Total training time: {:.2f}m'.format((time.time() - training_start_time) / 60))
    return min_loss, best_weights, loss_hist_train, loss_hist_valid
