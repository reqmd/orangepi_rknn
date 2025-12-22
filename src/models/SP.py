import torch.nn as nn

class SP(nn.Module):
    def __init__(self, input_size = 8 * 8 * 3, num_classes = 2):
        super(SP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        

    def forward(self, X):
        out = self.fc(X)
        out = self.relu(out)
        return out