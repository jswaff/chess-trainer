import torch

class Model(torch.nn.Module):

    def __init__(self, load_file=None):
        super(Model, self).__init__()

        N_L1_OUT = 256
        self.fc1 = torch.nn.Linear(768, N_L1_OUT)
        self.fc2 = torch.nn.Linear(N_L1_OUT*2, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.functional.relu

        if load_file is not None:
            print(f'loading model state from {load_file}')
            data = torch.load(load_file)
            #self._model.load_state_dict(data['model_state_dict'])
            self.load_state_dict(data['model_state_dict'])


    def forward(self, X1, X2):
        X1 = self.fc1(X1)
        X2 = self.fc1(X2)
        X = torch.cat((X1, X2), 1)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.fc3(X)
        return X
