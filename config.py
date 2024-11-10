import torch
from model import Model


class CFG:
    data_path = '/home/james/data/chess/labeled/rr.csv'
    input_model_name = None
    output_model_name = '/home/james/data/chess/models/d0-768-256-1.pt'

    num_features = 768
    batch_size = 1024 * 4
    num_workers = 8
    num_epochs = 1000
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_features, device, input_model_name)

    show_plots = True
