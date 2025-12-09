import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train import train_objective, train_final

class CNN(nn.Module):
    def __init__(self, num_classes = 2, n_filters = 16, dropout_p = 0.33, hidden_size = 256):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    def objective(self, trail, train_data, val_data, epochs, num_classes):
        lr = trail.suggest_float('lr', 5e-6, 1e-3, log=True)
        batch_size = trail.suggest_int('batch_size', 4, 16)

        n_filters = trail.suggest_int('n_filters', 8, 32)
        hidden_size = trail.suggest_int('hidden_size', 128, 1024)
        dropout_p = trail.suggest_float('dropout_p', 0, 1)

        model = CNN(num_classes=num_classes, n_filters=n_filters, hidden_size=hidden_size, dropout_p=dropout_p).to(self.device)
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
                               #2 - заморозили всю свертку
                               #3 - замораживаем только первый сверточный слой
                 
        model = CNN(num_classes=num_classes, 
                    n_filters = params['n_filters'], 
                    dropout_p=params['dropout_p'], 
                    hidden_size=params['hidden_size']).to(self.device)
        
        if freeze_param == 1:
            #загрузили веса
            model.load_state_dict(model_name)
            
            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили последний
            for param in model.fc2.parameters():
                param.requires_grad = True

            #переводим ВСЕ BN в Eval для того, чтобы не искажал данные
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    print(f'Eval OK {module}')
                    module.eval()
            
        if freeze_param == 2:
            #загрузили веса
            model.load_state_dict(model_name)
            
            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили fc1, fc2
            for param in model.fc2.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True

            #переводим ВСЕ BN в Eval для того, чтобы не искажал данные
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    print(f'Eval OK {module}')
                    module.eval()

        if freeze_param == 3:
            #загрузили веса
            model.load_state_dict(model_name)
            
            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили fc1, fc2
            for param in model.fc2.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True

            #разморозили последнюю свертку
            for param in model.layer2.parameters():
                param.requires_grad = True

            #переводим ПЕРВЫЙ BN в Eval для того, чтобы не искажал данные
            for name, module in model.batchnorm1.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    print(f'Eval OK {module}')
                    module.eval()

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