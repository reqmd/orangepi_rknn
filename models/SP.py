import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import train_objective, train_final

class SP(nn.Module):
    def __init__(self, input_size = 8 * 8 * 3, num_classes = 2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(SP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        

    def forward(self, X):
        out = self.fc(X)
        out = self.relu(out)
        return out

    def objective(self, trail, train_data, val_data, epochs, num_classes):
        lr = trail.suggest_float('lr', 5e-6, 1e-3, log=True)
        batch_size = trail.suggest_int('batch_size', 4, 16)

        model = SP(num_classes=num_classes).to(self.device)
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
                                device = self.device
                                )
        return loss
    
    def new_class(self, params, num_classes, epochs, train_data, val_data):
        model = SP(num_classes=num_classes).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr = params['lr'] * 0.1, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

        f1_best = train_final(train_data=train_data,
                            train_loader=train_loader,
                            valid_loader=val_loader,
                            model = model,
                            optim = optim,
                            loss_fn = loss_fn,
                            epochs=epochs,
                            device = self.device
                            )
        return f1_best