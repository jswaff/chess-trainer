import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from data import build_data_loaders
from train import train


def main():
    # torch.multiprocessing.set_start_method('spawn')
    print("device: ", CFG.device)

    train_dl, test_dl, valid_dl = build_data_loaders()

    # define the model
    model = nn.Sequential(
        nn.Linear(CFG.num_features, 2048),
        nn.ReLU(),
        nn.Linear(2048, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(CFG.device)

    # loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

    # train
    torch.manual_seed(1)
    hist = train(model, CFG.num_epochs, train_dl, valid_dl, loss_fn, optimizer)
    print(f'Min validation loss: {hist[0]:.4f}')
    # print(f'Best validation weights: {hist[1]}')

    # measure performance against test set
    loss_test = 0
    traced = False
    for x_batch, y_batch in test_dl:
        x_batch = x_batch.squeeze(0).to(CFG.device)
        y_batch = y_batch.squeeze(0).to(CFG.device)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss_test += loss.item()
        if not traced:
            model.load_state_dict(hist[1])
            traced_script_module = torch.jit.trace(model, x_batch)
            traced_script_module.save(CFG.model_name.replace(".pt", "-ts.pt"))
            traced = True
    loss_test /= len(test_dl)
    print(f'Test loss: {loss_test:.4f}')

    # visualize learning curves
    if CFG.show_plots:
        x_arr = np.arange(len(hist[2])) + 1
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist[2], '-o', label='Train loss')
        ax.plot(x_arr, hist[3], '--<', label='Validation loss')
        ax.legend(fontsize=15)
        plt.show()


if __name__ == "__main__":
    main()
