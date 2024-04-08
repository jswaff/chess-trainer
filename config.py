import torch

class CFG:

    input_path = 'data/pst-tapered.csv'
    model_name = 'models/pst-tapered-test.pt'
    num_samples = 1024 * 10

    num_features = 768
    batch_size = 256
    num_workers = 8
    num_epochs = 25
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    show_plots = True
