import torch.nn as nn

class EDGEAI28(nn.Module):
    def __init__(self, num_classes = 2, n_filters = 32, dropout_p = 0.33, hidden_size = 256):
        super().__init__()
        self.layer1 = nn.Conv2d(3, n_filters, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout2d(dropout_p)

        self.layer2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d((2, 2))      
        self.dropout2 = nn.Dropout2d(dropout_p)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * n_filters, hidden_size)
        self.relufc = nn.ReLU()
        self.dropoutfc = nn.Dropout1d(dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out = self.layer1(X)
        out = self.relu1(out)
        out = self.pooling1(out)
        out = self.dropout1(out)

        out = self.layer2(out)
        out = self.relu2(out)
        out = self.pooling2(out)
        out = self.dropout2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relufc(out)
        out = self.dropoutfc(out)
        return self.fc2(out)