import torch

class CFG:

    input_path = 'data/prophet-4_3.csv'
    model_name = 'models/prophet-4_3.pt'
    num_samples = None

    num_features = 768
    batch_size = 256
    num_workers = 8
    num_epochs = 100
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    show_plots = True
