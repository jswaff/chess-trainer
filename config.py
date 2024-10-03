import torch

from model import Model


class CFG:
    test_name = 'bfd'
    input_path = 'data/' + test_name + '.csv'
    #model_name = 'models/' + test_name + '.pt'
    model_name = 'models/hce-768-256-64-1.pt'

    num_features = 768
    batch_size = 512
    num_workers = 8
    num_epochs = 100
    lr = 0.001

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_features, device)

    show_plots = True
