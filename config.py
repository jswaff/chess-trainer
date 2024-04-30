import torch

class CFG:

    test_name = 'rr3-2d'
    input_path = 'data/' + test_name + '.csv'
    model_name = 'models/' + test_name + '.pt'
    num_samples = None

    num_features = 768
    batch_size = 512
    num_workers = 8
    num_epochs = 100
    lr = 0.001

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    show_plots = True
