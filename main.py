import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import optuna
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import LabeledDataset, TrainTestSubset, solve_imbalance
from utils.funcs import choise_model, match_case, save_yaml
from train import train_final
from test import test

labeled_root = r'C:\Users\Куликов\rice_dataset\valid'
pseudo_root = r'C:\Users\Куликов\rice_dataset\not_stratified_data'

data = LabeledDataset(labeled_root)
indices = list(range(len(data)))
train_indices, val_indices = train_test_split(indices, test_size=0.5, random_state=42, stratify=data.labels)

H, W = [], []
for images, _ in data:
    H.append(images.size[0])
    W.append(images.size[1])

size_0 = int(np.mean(H))
size_1 = int(np.mean(W))

model_name = 'best_model.pth'
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
    train_sampler = solve_imbalance(train_data)

    val_data = TrainTestSubset(data, val_indices, val_transform)
    val_sampler = solve_imbalance(val_data)

    epochs = 25
    num_classes = len(data.classes)
    n_trials = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    objective = partial(models[n].objective, train_data = train_data, val_data = val_data, num_classes = num_classes, epochs = epochs)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.CmaEsSampler())
    study.optimize(objective, n_trials=n_trials)
    params = study.best_params

    params['resolution'] = resolutions[n]
    params['num_classes'] = num_classes
    params['device'] = device
    model = match_case(params)

    print('///////////////////////////////////////')
    print(params)
    
    #подготовка параметров к тесту после подбора гиперпараметров
    optim = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], sampler=val_sampler)

    f1_threshhold = 0.98

    f1_best_before, f1_best_after = train_final(train_data=train_data,
                            train_loader=train_loader,
                            val_data = val_data,
                            val_loader=val_loader,
                            model = model,
                            params = params,
                            epochs=epochs,
                            model_name=model_name,
                            pseudo_labeling=pseudo_root
                            )
    if f1_threshhold < f1_best_before and f1_best_before > f1_best_after:
        print(f'Получена лучшая модель до псевдоразметки с параметрами: {params}, с лучшей метрикой {f1_best_before:.6f} ')
        break
    elif f1_threshhold < f1_best_after and f1_best_before < f1_best_after:
        print(f'Получена лучшая модель после псевдоразметки с параметрами: {params}, с лучшей метрикой {f1_best_after:.6f} ')
        break
    else:
        print('Точность у обоих методов одинаковая')
        break

save_yaml(params = params)

pseudo_model = match_case(params)
pseudo_model.load_state_dict(torch.load('pseudo_model.pth', weights_only=True))

model = match_case(params)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

f1, acc = test(val_loader=val_loader, params=params, model=pseudo_model)
print('Результат после псевдо разметки')
print(f'Accuracy: {acc}, F1 {f1}\n')

f1, acc = test(val_loader=val_loader, params=params, model=model)
print('Результат до псевдо разметки')
print(f'Accuracy: {acc}, F1 {f1}\n')