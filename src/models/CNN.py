import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes = 2, n_filters = 16, dropout_p = 0.33, hidden_size = 256):
        super().__init__()
        self.layer1 = nn.Conv2d(3, n_filters, kernel_size=(3, 3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(n_filters)
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride = 1, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(n_filters)
        self.pooling2 = nn.MaxPool2d((2, 2))
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * n_filters, hidden_size)
        self.relufc = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

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
        out = self.relufc(out)
        out = self.dropout(out)
        return self.fc2(out)