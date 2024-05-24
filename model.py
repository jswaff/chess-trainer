import torch.nn as nn


class Model:

    def __init__(self, num_features, device="cpu"):
        self._model = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

    @property
    def model(self):
        return self._model

