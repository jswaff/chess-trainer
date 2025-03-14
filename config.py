import torch

class CFG:
    data_path = 'data/labeled/positions-d5.csv'
    input_model_name = None
    output_model_name = 'models/nn-d5.pt'

    batch_size = 1024 * 16
    num_workers = 8
    num_epochs = 100
    lr = 0.001
    early_stop_counter = 3
    early_stop_threshold = 0.001
    Q = 127 / 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clear_cache = False
    show_plots = True
