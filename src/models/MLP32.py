import torch.nn as nn

class MLP32(nn.Module):
    def __init__(self, input_size = 32 * 32 * 3, hidden_size = 256, num_classes = 2):

        super(MLP32, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu2 = nn.ReLU()

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return self.relu2(out)
    
    