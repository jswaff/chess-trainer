import torch
from model import Model


class CFG:
    test_name = 'bfd-d0'
    input_path = '/home/james/data/chess/labeled/' + test_name + '.csv'
    model_name = '/home/james/data/chess/models/d0-768-256-1.pt'

    num_features = 768
    batch_size = 1024
    num_workers = 8
    num_epochs = 100
    lr = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = Model(num_features, device, model_name)
    model = Model(num_features, device)

    show_plots = True
