import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from data import build_data_loaders, clear_cache_dir
from model import Model
from train import train
from visualize import plot_learning_curves

def main():
    print("device: ", CFG.device)
    print("input model: ", CFG.input_model_name)
    print("output model: ", CFG.output_model_name)

    if CFG.clear_cache:
        print("clearing cache")
        clear_cache_dir()

    train_dl, test_dl, valid_dl = build_data_loaders()

    model = Model(CFG.input_model_name).to(CFG.device)

    # loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=1e-4)

    # train
    torch.manual_seed(1)
    min_loss, best_weights, loss_hist_train, loss_hist_valid = train(model, CFG.num_epochs, train_dl, valid_dl, loss_fn, optimizer)
    print(f'Min validation loss: {min_loss:.4f}')

    # measure performance against test set
    model.load_state_dict(best_weights)
    loss_test = 0
    traced = False
    for x_batch, x2_batch, y_batch in test_dl:
        x_batch = x_batch.to(CFG.device)
        x2_batch = x2_batch.to(CFG.device)
        y_batch = y_batch.to(CFG.device)
        pred = model(x_batch, x2_batch)
        loss = loss_fn(pred, y_batch)
        loss_test += loss.item()
        # trace model and save in torch script format
        if not traced:
            model.to("cpu")
            traced_script_module = torch.jit.trace(model, (x_batch.to_dense().to("cpu"), x2_batch.to_dense().to("cpu")))
            traced_script_module.save(CFG.output_model_name.replace(".pt", "-ts.pt"))
            model.to(CFG.device)
            traced = True
    loss_test /= len(test_dl)
    print(f'Test loss: {loss_test:.4f}')

    # visualize learning curves
    if CFG.show_plots:
        plot_learning_curves(loss_hist_train, loss_hist_valid)


if __name__ == "__main__":
    main()
