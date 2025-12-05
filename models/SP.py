import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train_objective

class SP(nn.Module):
    def __init__(self, input_size = 8 * 8 * 3, num_classes = 2):
        super(SP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.fc(X)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def objective(self, trail, train_data, val_data, epochs, num_classes):
        lr = trail.suggest_float('lr', 5e-6, 1e-3, log=True)
        batch_size = trail.suggest_int('batch_size', 4, 16)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SP(num_classes=num_classes).to(device)
        
        optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        loss = train_objective(train_data=train_data,
                                train_loader=train_loader,
                                valid_loader=val_loader,
                                model = model,
                                optim = optim,
                                loss_fn = loss_fn,
                                epochs=epochs,
                                device = device
                                )
        return loss