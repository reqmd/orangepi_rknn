from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import optuna
import time

from src.data.dataset import LabeledDataset
from src.data.funcs import dataset_into_loader, train_test_split
from src.models.SP import SP
from src.models.MLP16 import MLP16
from src.models.MLP32 import MLP32
from src.models.CNN import CNN
from src.models.MobileNet import MobileNet
from src.utils.config_funcs import load_yaml, save_yaml
from src.utils.early_stopping import EarlyStopping
from src.training.train import __train__

def __hyperparams__(data, mode_name, f1_threshhold = 0.98, test_size = 0.5):
    num_classes = len(data.classes)
    prep_params = load_yaml('configs/static/prep_configs/st_prep_hyperparams_config.yaml')
    model_params = load_yaml('configs/dynamic/model_configs/mnd_config.yaml')

    for idx, resolution in enumerate(model_params['resolutions']):
        train_data, val_data = train_test_split(data, resolution=resolution, test_size=test_size)
        model_name = model_params['models'][idx]
        print(model_name, resolution)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, 
                                               model_name=model_name, 
                                               num_classes=num_classes, 
                                               train_data=train_data, 
                                               val_data=val_data), n_trials=prep_params['n_trials'])
        best_params = study.best_params
        save_best_params(best_params=best_params, model_name=model_name, num_classes=num_classes, resolution=resolution)
        f_time = time.strftime("%d.%m_%H.%M")
        DATA_PATH = './data'
        modename_path = os.path.join(DATA_PATH, mode_name)
        model_name = f'{f_time}-{mode_name}.pth'

        f1_best = __train__(data = data, model_name=model_name, modename_path=modename_path, use_for_hyperparams=True)
        
        if f1_best > f1_threshhold:
            break

def objective(trial: optuna.Trial, model_name: str, num_classes: int, train_data: LabeledDataset, val_data: LabeledDataset):
    #Гиперпараметры для моделей должны сохранены СТРОГО в следующем порядке: lr, weight_decay, model_params
    params = load_yaml('configs/static/model_configs/st_hyperparams_config.yaml')
    prep_params = load_yaml('configs/static/prep_configs/st_prep_hyperparams_config.yaml')
    device = prep_params['device']
    model_params = params['Models'][model_name]
    batch_size = trial_config(trial = trial, param = params['Base'])

    lr = trial_config(trial = trial, param = {'lr':model_params['lr']})
    weight_decay = trial_config(trial = trial, param = {'weight_decay':model_params['weight_decay']})
    #перебор моделей параметров
    trials_model = {}
    for param_key in model_params:
        if param_key == 'lr' or param_key == 'weight_decay':
            continue
        else:
            trials_model[param_key] = (trial_config(trial = trial, param = {param_key:model_params[param_key]}))

    model = match_case(model_name=model_name, trials_model=trials_model, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay) #0.01 - стандартное значение

    train_loader, val_loader = dataset_into_loader(data=[train_data, val_data], batch_size=batch_size)

    loss = train_objective(train_loader=train_loader, val_loader=val_loader, model=model, optim=optim, loss_fn=loss_fn)
    return loss

def train_objective(train_loader: DataLoader, val_loader: DataLoader,  model, optim, loss_fn):
    params = load_yaml('configs/static/prep_configs/st_prep_hyperparams_config.yaml')
    epochs = params['epochs']
    device = params['device']
    estop = EarlyStopping(min_delta=0.01)
    total_loss = []

    print('Начат цикл обучения:')
    for epoch in range(epochs):

        train_loss = []
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.cpu().detach().numpy())
    
        
        test_loss = []
        y_preds = []
        y_trues = []

        model.eval()
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y_trues.extend(y.cpu().numpy())
            y_pred = model(X)
            val_loss = loss_fn(y_pred, y)
            test_loss.append(val_loss.cpu().detach().numpy())
            y_pred = torch.argmax(y_pred, dim=1)
            y_preds.extend(y_pred.cpu().numpy())
        total_loss.append(np.mean(test_loss))
        
        if estop.step(loss_now = np.mean(test_loss)) == 1: #ранняя остановка
            print(f'Эпоха: {epoch+1}')
            break
        
    f1_sc = f1_score(y_trues, y_preds, average='weighted')
    print(f'F1: {f1_sc:.6f}, Test Loss {np.mean(test_loss)}\n')
    return total_loss[-1]

def trial_config(trial: optuna.Trial, param: dict):
    #param: dict ->  {'param_name':['float'/'int', start, stop, step]}
    param_name = list(param.keys())[0]
    #print(param[param_name][1:])
    if param[param_name][0] == 'float':
        hyperparam = trial.suggest_float(param_name, **param[param_name][1])
        hyperparam = np.round(hyperparam, 5)

    elif param[param_name][0] == 'int':
        hyperparam = trial.suggest_int(param_name, **param[param_name][1])

    else:
        print('В КОНФИГУРАЦИИ ЗАГРУЖЕНА КАТЕГОРИАЛЬНАЯ ПЕРЕМЕННАЯ, ЧТО НЕДОПУСТИМО')

    return hyperparam

def match_case(model_name: str, trials_model: dict, num_classes: int):
    match model_name:
        case 'SP':
            model = SP(num_classes=num_classes, **trials_model)
        case 'MLP16':
            model = MLP16(num_classes=num_classes, **trials_model)
        case 'MLP32':
            model = MLP32(num_classes=num_classes, **trials_model)
        case 'CNN':
            model = CNN(num_classes=num_classes, **trials_model)
        case 'MobileNet':
            model = MobileNet(num_classes=num_classes, **trials_model)
    return model

def save_best_params(best_params: dict, model_name: str, num_classes: int, resolution: int):
    best_params['resolution'] = resolution
    best_params['num_classes'] = num_classes
    best_params['model'] = model_name
    for param in best_params:
        if isinstance(best_params[param], float):
            best_params[param] = round(best_params[param], 5)

    save_yaml('configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml', best_params)
    print(f'Сохранены следующий параметры: {best_params}')