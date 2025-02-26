import torch
from config import CFG

NN_SIZE_L1 = 1536
NN_SIZE_L2 = 1

class Model(torch.nn.Module):


    def __init__(self, load_file=None):
        super(Model, self).__init__()

        self.fc1 = torch.nn.Linear(768, NN_SIZE_L1)
        self.fc2 = torch.nn.Linear(NN_SIZE_L1 * 2, NN_SIZE_L2)

        if load_file is not None:
            print(f'loading model state from {load_file}')
            data = torch.load(load_file)
            self.load_state_dict(data['model_state_dict'])


    def forward(self, X1, X2):
        X1 = self.fc1(X1)
        X2 = self.fc1(X2)
        X = torch.cat((X1, X2), axis=1)

        X = torch.clamp(X, min=0.0, max=CFG.Q)

        X = self.fc2(X)
        X = torch.clamp(X, min=-CFG.Q, max=CFG.Q)

        return X
