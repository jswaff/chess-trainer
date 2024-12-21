import torch
from model import Model


class CFG:
    data_path = '/home/james/data/chess/labeled/positions-d2.csv'
    input_model_name = None #'/home/james/data/chess/models/all-d1.pt'
    output_model_name = '/home/james/data/chess/models/nn-004-d2.pt'

    batch_size = 1024
    num_workers = 8
    num_epochs = 300
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(input_model_name)

    show_plots = True
