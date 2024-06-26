import torch
import torch.nn as nn


class Model:

    def __init__(self, num_features, device="cpu", load_file=None):
        self._model = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        if load_file is not None:
            print(f'loading model state from {load_file}')
            data = torch.load(load_file)
            self._model.load_state_dict(data['model_state_dict'])

    @property
    def model(self):
        return self._model

