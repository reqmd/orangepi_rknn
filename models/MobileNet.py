import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train import train_objective, train_final

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1 if stride == 1 else 0,
                groups=in_channels
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)

class MobileNet(nn.Module):
    def __init__(self, alpha=1, num_classes=2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.layer_0 = nn.Conv2d(
            3,
            int(32 * self.alpha),
            stride=2,
            kernel_size=(3, 3),
            padding=1
        )
        self.layer_1 = self.__make_layer(int(32 * self.alpha), int(64 * self.alpha), 1)
        self.layer_2 = self.__make_layer(int(64 * self.alpha), int(128 * self.alpha), 1)
        self.layer_3 = self.__make_layer(int(128 * self.alpha), int(128 * self.alpha), 2)  # Даунсэмплинг
        self.layer_4 = self.__make_layer(int(128 * self.alpha), int(256 * self.alpha), 1)
        self.layer_5 = self.__make_layer(int(256 * self.alpha), int(256 * self.alpha), 2)  # Даунсэмплинг
        self.layer_6 = self.__make_layer(int(256 * self.alpha), int(512 * self.alpha), 1)
        self.layer_7 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 2)  # Даунсэмплинг
        self.layer_8 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_9 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_10 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_11 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_12 = self.__make_layer(int(512 * self.alpha), int(1024 * self.alpha), 1)
        self.layer_13 = self.__make_layer(int(1024 * self.alpha), int(1024 * self.alpha), 2)  # Даунсэмплинг
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(1024 * self.alpha), num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        return MobileNetBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, X):
        out = self.layer_0(X)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.layer_10(out)
        out = self.layer_11(out)
        out = self.layer_12(out)
        out = self.layer_13(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return self.fc(out)

    def objective(self, trail, train_data, val_data, epochs, num_classes):
        lr = trail.suggest_float('lr', 5e-6, 1e-3, log=True)
        batch_size = trail.suggest_int('batch_size', 4, 16)

        alpha = trail.suggest_float('alpha', 0, 1)

        model = MobileNet(num_classes=num_classes, alpha = alpha).to(self.device)
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
                               #2 - замораживаем все, кроме fc и 2-х последних сверточных блоков
                 
        model = MobileNet(num_classes=num_classes, alpha=params['alpha']).to(self.device)
        if freeze_param == 1:
            #загрузили веса
            model.load_state_dict(model_name)

            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили последний
            for param in model.fc.parameters():
                param.requires_grad = True

            #переводим ВСЕ BN в Eval
            for name, module in model.named_modules():
                for name1, module1 in module.named_modules():
                    if isinstance(module1, torch.nn.BatchNorm2d):
                        print(f'Eval OK {module1}')
                        module1.eval()

        if freeze_param == 2:
            #загрузили веса
            model.load_state_dict(model_name)

            #заморозили все слои
            for param in model.parameters():
                param.requires_grad = False

            #разморозили последний
            for param in model.fc.parameters():
                param.requires_grad = True

            #размораживаем последний сверточный блок
            for param in model.layer_13.parameters():
                param.requires_grad = True

            #переводим ВСЕ BN в Eval
            for name, module in model.named_modules():
                for name1, module1 in module.named_modules():
                    if isinstance(module1, torch.nn.BatchNorm2d):
                        print(f'Eval OK {module1}')
                        module1.eval()
            
            #переводим последние BN из layer13 в Train
            for name, module in model.layer_13.named_modules():
                for name1, module1 in module.named_modules():
                    if isinstance(module1, torch.nn.BatchNorm2d):
                        print(f'Train OK {module1}')
                        module1.eval()

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