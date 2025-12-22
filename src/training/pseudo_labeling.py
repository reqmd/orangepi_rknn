import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score

from src.data.funcs import dataset_into_loader, train_test_split
from src.utils.save_load import load_model, save_model
from src.utils.config_funcs import load_yaml
from src.data.transforms import return_transforms
from src.data.dataset import LabeledDataset, TrainTestSubset
from src.utils.early_stopping import EarlyStopping

def __pseudo_labeling__(data,
                        new_data_root, 
                        model_name = 'standart_model.pth', 
                        model_config_root = 'configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml',
                        prep_config_root = 'configs/static/prep_configs/st_prep_pseudolabel.yaml'):
    print('Начало псведо разметки')
    prep_params = load_yaml(prep_config_root)

    n_iter = prep_params['n_iter']
    n_epochs = prep_params['n_epochs']
    device = prep_params['device']
    threshhold = prep_params['threshhold']

    model_params = load_yaml(model_config_root)
    model = load_model(model_params, model_name=model_name).to(device)
    all_params = load_yaml(model_config_root)

    batch_size = all_params['resolution']

    train_data, val_data = train_test_split(data, batch_size)
    _, val_loader = dataset_into_loader(data = [train_data, val_data], batch_size=batch_size)

    best_loss = float('inf')
    best_state_dict = None
    model_name = 'new_model.pth'
    new_data, new_loader = new_data_get(unlabeled_root=new_data_root, batch_size=batch_size, resolution=all_params['resolution'])
    for itr in range(n_iter):
        print(f'Iter: {itr+1}/{n_iter}')
        datalen = len(new_data)

        new_labeled_data, indices = new_data_generation(model=model, loader=new_loader, device=device, datalen=datalen, threshhold=threshhold)
        combined_data = new_data_concat(new_data=new_labeled_data, old_data=train_data, batch_size=batch_size)

        new_data = new_data_del(data=new_data, indices=indices)
        combined_loader = dataset_into_loader(data=combined_data, batch_size=batch_size)
        print(len(combined_data))
        best_loss, best_state_dict = new_data_train(train_loader=combined_loader,
                       val_loader=val_loader,
                       model=model,
                       params=all_params,
                       n_epochs=n_epochs,
                       device=device,
                       best_params=[best_loss, best_state_dict])
        
    print(f'В файл {model_name} была сохранена лучшая модель с наименьшим лоссом: {best_loss}')
    save_model(best_state_dict, model_name=model_name)

def new_data_get(unlabeled_root, resolution, batch_size):
    #загружает датасет неразмеченных данных и преобразует их в лоадер
    train_tranform, _ = return_transforms(resolutions=resolution)
    data = LabeledDataset(root = unlabeled_root, transform=train_tranform)
    loader = DataLoader(data, batch_size=batch_size)
    return data, loader

def new_data_generation(model, loader, device, threshhold, datalen):
    #возвращает новый набор данных и индексы на удаление фото которые найдены
    model.eval()
    pseudo_labeled_X = []
    pseudo_labeled_y = []
    indices = []
    for batch_idx, (X, _) in enumerate(loader):
        X = X.to(device)
        y_pred = model(X)
        probs = F.softmax(y_pred, dim = 1)
        conf, preds = torch.max(probs, 1) #процент уверенности, класс
        #print(conf, preds)
        for j in range(len(conf)):
            global_index = batch_idx * loader.batch_size + j
            if conf[j] > threshhold:
                pseudo_labeled_X.append(X[j].cpu())
                pseudo_labeled_y.append(preds[j].cpu())
            else:
                indices.append(global_index)
    print(f'Длина новых данных после псведоразметки: {len(pseudo_labeled_y)} / {datalen}, {len(pseudo_labeled_y) / datalen * 100:.2f}')
    print(len(indices))
    pseudo_labeled_X = torch.stack(pseudo_labeled_X)
    pseudo_labeled_y = torch.stack(pseudo_labeled_y)
    after_data = TensorDataset(pseudo_labeled_X, pseudo_labeled_y)
    if len(after_data) == 0:
        raise ValueError('Отсутсвуют данные для псевдоразметки!')
    return after_data, indices

def new_data_concat(new_data, old_data, batch_size):
    #соединяет новый и старые наборы данных
    labels = [label for _, label in new_data]
    class_counts = np.bincount(labels)
    print(f'Вывод распределения новых данных по классам: {class_counts}')
    combined_data = torch.utils.data.ConcatDataset([old_data, new_data])


    labels = [label for _, label in combined_data]
    class_counts = np.bincount(labels)
    print(f'Вывод распределения классов после соединения данных и распределения классов: {class_counts}')
    return combined_data

def new_data_del(data, indices):
    #собирает набор данных только из тех данных, которые не смогла обработать
    data = TrainTestSubset(data, indices=indices)
    return data

def new_data_train(train_loader, val_loader, model, params, n_epochs, device, best_params: list):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr = params['lr'] * 0.5, weight_decay=params['weight_decay'])
    estop = EarlyStopping(patience=3, min_delta=0.01)
    best_loss, best_state_dict = best_params

    model.train()
    for epoch in range(n_epochs):
        train_loss = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch: {epoch+1}/{n_epochs}, Loss: {np.mean(train_loss):.5f}')

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

        if np.mean(test_loss) < best_loss:
            best_loss = np.mean(test_loss)
            best_state_dict = model.state_dict()

        f1 = f1_score(y_trues, y_preds, average='weighted')
        print(f'F1: {f1:.6f}, Test Loss {np.mean(test_loss)}\n')

        if estop.step(np.mean(test_loss)) == 1:
            break
    return best_loss, best_state_dict