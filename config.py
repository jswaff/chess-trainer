import torch

from model import Model


class CFG:
    test_name = 'E12.52-1M-D12'
    input_path = 'data/' + test_name + '.csv'
    #model_name = 'models/' + test_name + '.pt'
    model_name = 'models/test.pt'

    num_features = 768
    batch_size = 512
    num_workers = 8
    num_epochs = 100
    lr = 0.001

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_features, device)

    show_plots = True
