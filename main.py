import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from data import build_data_loaders
from train import train
from visualize import plot_learning_curves


def main():
    print("device: ", CFG.device)
    print("input model: ", CFG.input_model_name)
    print("output model: ", CFG.output_model_name)

    train_dl, test_dl, valid_dl = build_data_loaders()

    model = CFG.model.to(CFG.device)

    # loss function and optimizer
    loss_fn = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    optimizer = optim.Adagrad(model.parameters(), lr=CFG.lr)

    # train
    torch.manual_seed(1)
    hist = train(model, CFG.num_epochs, train_dl, valid_dl, loss_fn, optimizer)
    print(f'Min validation loss: {hist[0]:.4f}')

    # measure performance against test set
    loss_test = 0
    traced = False
    for x_batch, x2_batch, y_batch in test_dl:
        x_batch = x_batch.squeeze(0).to(CFG.device)
        x2_batch = x2_batch.squeeze(0).to(CFG.device)
        y_batch = y_batch.squeeze(0).to(CFG.device)
        pred = model(x_batch, x2_batch)
        loss = loss_fn(pred, y_batch)
        loss_test += loss.item()
        if not traced:
            model.load_state_dict(hist[1])
            model.to("cpu")
            traced_script_module = torch.jit.trace(model, (x_batch.to("cpu"), x2_batch.to("cpu")))
            traced_script_module.save(CFG.output_model_name.replace(".pt", "-ts.pt"))
            model.to(CFG.device)
            traced = True
    loss_test /= len(test_dl)
    print(f'Test loss: {loss_test:.4f}')

    # visualize learning curves
    if CFG.show_plots:
        plot_learning_curves(hist[2], hist[3])


if __name__ == "__main__":
    main()
