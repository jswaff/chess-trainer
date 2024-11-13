import torch

class Model(torch.nn.Module):

    def __init__(self, num_features, load_file=None):
        super(Model, self).__init__()

        self.fc1 = torch.nn.Linear(num_features, 768)
        self.fc2 = torch.nn.Linear(768, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.functional.relu

        if load_file is not None:
            print(f'loading model state from {load_file}')
            data = torch.load(load_file)
            #self._model.load_state_dict(data['model_state_dict'])
            self.load_state_dict(data['model_state_dict'])


    def forward(self, X):
        X = self.fc1(X)
        X = self.relu(X)
        #torch.clamp(X, 0, 1)
        X = self.fc2(X)
        X = self.relu(X)
        #torch.clamp(X, 0, 1)
        X = self.fc3(X)
        return X
