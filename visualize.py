import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from config import CFG


def plot_learning_curves(loss_hist_train, loss_hist_valid):
    epochs = len(loss_hist_train)
    x_arr = np.arange(len(loss_hist_train)) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, loss_hist_train, '-o', label='Train loss')
    ax.plot(x_arr, loss_hist_valid, '--<', label='Validation loss')
    model_name = os.path.splitext(os.path.basename(CFG.model_name))[0]
    ax.set_title(f'{model_name} {epochs} epochs')
    ax.legend(fontsize=15)
    plt.show()


if __name__ == "__main__":
    data = torch.load(CFG.model_name)
    e = data['epoch']
    lht = data['loss_hist_train'][0:e]
    lhv = data['loss_hist_valid'][0:e]
    plot_learning_curves(lht, lhv)
