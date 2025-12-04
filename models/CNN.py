import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes = 2, hidden_size = 16):
        super().__init__()
        self.layer1 = nn.Conv2d(3, hidden_size, kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride = 1, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size)
        self.pooling2 = nn.MaxPool2d((2, 2))
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * hidden_size, 256)
        self.dropout = nn.Dropout1d(0.33)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, X):
        out = self.layer1(X)
        out = self.batchnorm1(out)
        out = self.pooling1(out)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.batchnorm2(out)
        out = self.pooling2(out)
        out = self.relu2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)
