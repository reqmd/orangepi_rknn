import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from src.utils.config_funcs import load_yaml
from src.data.dataset import LabeledDataset
from src.data.funcs import train_test_split, dataset_into_loader
from src.utils.early_stopping import EarlyStopping
from src.utils.save_load import save_model, load_model

def __train__(data,
              use_for_hyperparams = False):
    prep_params = load_yaml('configs/static/prep_configs/st_prep_config.yaml')
    model_params = load_yaml('configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml')
    model = load_model(params=model_params).to('cpu')
    all_params = load_yaml('configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml')

    if isinstance(data, LabeledDataset):
        train_data, val_data = train_test_split(data, resolution=all_params['resolution']) #если подали неразделенный датасет
    else:
        train_data, val_data = data #если дали разделенные датасеты в списке

    train_loader, val_loader = dataset_into_loader(data=[train_data, val_data], batch_size=all_params['batch_size'])
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr = all_params['lr'], weight_decay=all_params['weight_decay'])
    device = prep_params['device']
    print(device)
    epochs = prep_params['epochs']
    estop = EarlyStopping(min_delta=0.01)

    total_loss = []
    f1_best = 0

    print('Начат цикл обучения:')
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X).to(device)
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {np.mean(train_loss)}')
    
        test_loss = []
        y_preds = []
        y_trues = []
        model.eval()
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            y_trues.extend(y.cpu().numpy())
            y_pred = model(X).to(device)
            val_loss = loss_fn(y_pred, y)
            test_loss.append(val_loss.cpu().detach().numpy())
            y_pred = torch.argmax(y_pred, dim=1)
            y_preds.extend(y_pred.cpu().numpy())

        total_loss.append(np.mean(test_loss))
        f1 = f1_score(y_trues, y_preds, average='weighted')
        print(f'F1: {f1:.6f}, Test Loss {np.mean(test_loss)}\n')

        if estop.step(np.mean(test_loss)) == 1:
            break
        
        if f1 >= f1_best:
            f1_best = f1
            f1_best_epoch = epoch
            model_state_dict = model.state_dict()

    save_model(model_state_dict, 'standart_model.pth')
    print(f'Лучшая метрика была достигнута на {f1_best_epoch+1} эпохе, значение f1 {f1_best:.6f}')
    if use_for_hyperparams == True:
        return f1_best
    
