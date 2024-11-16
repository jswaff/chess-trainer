import torch
from model import Model


class CFG:
    data_path = '/home/james/data/chess/labeled/all-d1.csv'
    input_model_name = None
    output_model_name = '/home/james/data/chess/models/all-d1.pt'

    batch_size = 1024 * 32
    num_workers = 8
    num_epochs = 1000
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(input_model_name)

    show_plots = True
