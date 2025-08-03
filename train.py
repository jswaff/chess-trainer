import copy
import time
import numpy as np
import torch

from config import CFG
from data import save_model


def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    min_loss = np.inf
    best_weights = copy.deepcopy(model.state_dict())
    training_start_time = time.time()
    orig_early_stop_cnt = CFG.early_stop_counter
    early_stop_cnt = orig_early_stop_cnt
    print('Epoch        Train        Valid        ETime        TTime        Delta        ESCnt')
    print('-----------------------------------------------------------------------------------')
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        for x_batch, x2_batch, y_batch, y2_batch in train_dl:
            x_batch = x_batch.to(CFG.device)
            x2_batch = x2_batch.to(CFG.device)
            y_batch = y_batch.to(CFG.device)
            y2_batch = y2_batch.to(CFG.device)
            y1_pred, y2_pred  = model(x_batch, x2_batch)
            loss = loss_fn(y1_pred, y_batch) + loss_fn(y2_pred, y2_batch)
            # TODO: factor in quantization error
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist_train[epoch] += loss.item()  # this is for an entire batch
            # clip weights
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-CFG.Q, CFG.Q)
        loss_hist_train[epoch] /= len(train_dl)

        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            for x_batch, x2_batch, y_batch, y2_batch in valid_dl:
                x_batch = x_batch.to(CFG.device)
                x2_batch = x2_batch.to(CFG.device)
                y_batch = y_batch.to(CFG.device)
                y2_batch = y2_batch.to(CFG.device)
                y1_pred, y2_pred = model(x_batch, x2_batch)
                loss = loss_fn(y1_pred, y_batch) + loss_fn(y2_pred, y2_batch)
                loss_hist_valid[epoch] += loss.item()
            loss_hist_valid[epoch] /= len(valid_dl)

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
                'loss_hist_train': loss_hist_train[0:epoch],
                'loss_hist_valid': loss_hist_valid[0:epoch],
            }, CFG.output_model_name)
            save_model(model, CFG.output_model_name.replace(".pt", ".txt"))
            save_model(model, CFG.output_model_name.replace(".pt", "-q.txt"), True)

        if delta >= -CFG.early_stop_threshold:
            early_stop_cnt = early_stop_cnt - 1
        else:
            early_stop_cnt = orig_early_stop_cnt

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

