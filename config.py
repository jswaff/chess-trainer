import torch
from model import Model


class CFG:
    #data_path = '/home/james/data/chess/labeled/positions_plus_ccrl-d1.csv'
    data_path = '/home/james/data/chess/labeled/positions-hce.csv'
    input_model_name = None # '/home/james/data/chess/models/hce-768-256-1.pt'
    #output_model_name = '/home/james/data/chess/models/positions_plus_ccrl-d1-768-256-1.pt'
    output_model_name = '/home/james/data/chess/models/positions-hce-768-256-1.pt'

    batch_size = 1024
    num_workers = 8
    num_epochs = 1000
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(input_model_name)

    show_plots = True
