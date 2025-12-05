import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import optuna
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LabeledDataset, TrainTestSubset
from funcs import calculate_avg_size_per_class, choise_model
from models import MLP, MobileNet, SP, CNN
from train import train_final

labeled_root = r'C:\Users\Куликов\rice_dataset\valid'
data = LabeledDataset(labeled_root)
#avg_sizes, (max_h, max_w) = calculate_avg_size_per_class(labeled_dataset, num_classes)
#print(max_h, max_w)
indices = list(range(len(data)))
train_indices, val_indices = train_test_split(indices, test_size=0.5, random_state=42, stratify=data.labels)

H, W = [], []
for images, _ in data:
    H.append(images.size[0])
    W.append(images.size[1])

size_0 = int(np.mean(H))
size_1 = int(np.mean(W))

resolutions, models = choise_model([size_0, size_1])
f1_threshhold = 0

for n in range(len(resolutions)):
    print('Полученное разрешение: ', resolutions[n])
    train_transform = transforms.Compose([
        transforms.Resize((resolutions[n], resolutions[n])), 
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std),
        #transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        #transforms.ColorJitter(0.1, 0.1, 0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((resolutions[n], resolutions[n])), 
        #transforms.Normalize(mean=mean, std=std),
        transforms.ToTensor()
    ])

    train_data = TrainTestSubset(data, train_indices, train_transform)
    val_data = TrainTestSubset(data, val_indices, val_transform)
    epochs = 25
    num_classes = len(data.classes)
    n_trials = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    objective = partial(models[n].objective, train_data = train_data, val_data = val_data, num_classes = num_classes, epochs = epochs)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.CmaEsSampler())
    study.optimize(objective, n_trials=n_trials)
    params = study.best_params
    match resolutions[n]:
        case 8:
            model = SP.SP(num_classes=num_classes).to(device)
        case 32:
            model = MLP.MLP(num_classes=num_classes, hidden_size=params['hidden_size']).to(device)
        case 64:
            model = CNN.CNN(num_classes=num_classes, n_filters = params['n_filters'], dropout_p=params['dropout_p'], hidden_size=params['hidden_size']).to(device)
        case 224:
            model = MobileNet.MobileNet(num_classes=num_classes, alpha=params['alpha']).to(device)

    print('///////////////////////////////////////')
    print(params)
    
    #подготовка параметров к тесту после подбора гиперпараметров
    optim = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
    f1_threshhold = train_final(train_data=train_data,
                            train_loader=train_loader,
                            valid_loader=val_loader,
                            model = model,
                            optim = optim,
                            loss_fn = loss_fn,
                            epochs=epochs,
                            device = device,
                            f1_threshhold = f1_threshhold
                            )
    print(f1_threshhold)
    if f1_threshhold > 0.98:
        print(f'Получена лучшая модель с разрешением {resolutions[n], }, параметрами: {params}, с лучшей метрикой {f1_threshhold:.6f}')
        break





