import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train import train_objective, train_final

class MLP(nn.Module):
    def __init__(self, input_size = 32 * 32 * 3, hidden_size = 256, num_classes = 2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(MLP, self).__init__()
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
    
    def objective(self, trail, train_data, val_data, epochs, num_classes):
        lr = trail.suggest_float('lr', 5e-6, 1e-3, log=True)
        batch_size = trail.suggest_int('batch_size', 4, 16)

        hidden_size = trail.suggest_int('hidden_size', 128, 1024)

        model = MLP(num_classes=num_classes,  hidden_size=hidden_size).to(self.device)
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
    
    def new_class(self, params, num_classes, epochs, train_data, val_data, freeze_param, model_name = 'best_model.pth'):
        #freeze_param: int -->  0 - ничего не замораживать, обучение по новой
                               #1 - НЕ замораживаем последний слой, остальное замораживаем
                 
        model = MLP(num_classes=num_classes, hidden_size=params['hidden_size']).to(self.device)
        if freeze_param == 1:
            #загрузили веса
            model.load_state_dict(model_name)

            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили последний
            for param in model.fc2.parameters():
                param.requires_grad = True

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