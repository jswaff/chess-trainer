import torch

class CFG:

    input_path = 'data/E12.33-1M-D12-Resolved.book-labeled.csv'
    model_name = 'models/E12.33-1M-D12-Resolved.book.pt'
    num_samples = None

    num_features = 768
    batch_size = 256
    num_workers = 8
    num_epochs = 100
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    show_plots = True
