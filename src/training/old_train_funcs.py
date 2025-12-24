import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import f1_score

from data.dataset import LabeledDataset, solve_imbalance, TrainTestSubset
from utils import funcs


def train_objective(
    train_loader, 
    val_loader,  
    model, 
    optim, 
    loss_fn, 
    device,
    epochs
):
    
    total_loss = []
    best_loss = float('inf')
    counter = 0

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
    
        model.eval()
        test_loss = []
        y_preds = []
        y_trues = []
        patience = 3
        
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

        if best_loss > np.mean(test_loss):
            best_loss = np.mean(test_loss)
            counter = 0 
        else:
            counter+=1
        if counter >= patience:
            print(f'Ранняя остановка на {epoch} эпохе')
            break

    print(f'F1: {f1_score(y_trues, y_preds, average='weighted'):.6f}, Test Loss {np.mean(test_loss)}\n')

    return total_loss[-1]



def train_final(
    train_data,
    train_loader,
    val_data, 
    val_loader,  
    model, 
    params,
    epochs,
    model_name = 'best_model.pth',
    pseudo_labeling = None
):
    
    total_loss = []
    best_loss = float('inf')
    counter = 0
    f1_best = 0
    f1_best_epoch = 0
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params = model.parameters(), lr = params['lr'], weight_decay=0.01)

    print('Начат цикл обучения:')
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for X, y in train_loader:
            X = X.to(params['device'])
            y = y.to(params['device'])
            y_pred = model(X).to(params['device'])
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {np.mean(train_loss)}')
    
        model.eval()
        test_loss = []
        y_preds = []
        y_trues = []
        patience = 3
        
        for X, y in val_loader:
            X = X.to(params['device'])
            y = y.to(params['device'])
            y_trues.extend(y.cpu().numpy())
            y_pred =  model(X).to(params['device'])
            val_loss = loss_fn(y_pred, y)
            test_loss.append(val_loss.cpu().detach().numpy())
            y_pred = torch.argmax(y_pred, dim=1)
            y_preds.extend(y_pred.cpu().numpy())
    
        
        total_loss.append(np.mean(test_loss))

        if best_loss > np.mean(test_loss):
            best_loss = np.mean(test_loss)
            counter = 0
            torch.save(model.state_dict(), model_name) 
        else:
            counter+=1
        if counter >= patience:
            print(f'Ранняя остановка на {epoch} эпохе')
            break
        f1 = f1_score(y_trues, y_preds, average='weighted')
        print(f'F1: {f1:.6f}, Test Loss {np.mean(test_loss)}\n')

        if f1 >= f1_best:
            f1_best = f1
            f1_best_epoch = epoch
            torch.save(model.state_dict(), model_name)

    print(f'Лучшая метрика была достигнута на {f1_best_epoch+1} эпохе, значение f1 {f1_best:.6f}')
    if pseudo_labeling != None:
        f1_best_pseudo = pseudo_label(
            model = model,
            params=params,
            pseudo_root = pseudo_labeling,
            train_data = train_data,
            val_data = val_data,
            threshhold=0.95,
            n_iter = 3,
            epochs_retrain=5,
        )
        print(np.round(f1_best, 6), np.round(f1_best_pseudo, 6))
        return f1_best, f1_best_pseudo
    else:
        return f1_best, 0



def pseudo_label(model,
                 params, 
                 pseudo_root, 
                 train_data,
                 val_data,   
                 threshhold = 0.95,
                 n_iter = 3,
                 epochs_retrain = 5
                ):
    print('Начало псведо разметки')
    pseudo_transform = transforms.Compose([
        transforms.Resize((params['resolution'], params['resolution'])), 
        transforms.ToTensor()
    ])
    pseudo_data = LabeledDataset(pseudo_root)
    pseudo_data = TrainTestSubset(pseudo_data, indices=range(len(pseudo_data)), transform=pseudo_transform)
    pseudo_sampler = solve_imbalance(pseudo_data)
    
    pseudo_loader = DataLoader(pseudo_data, sampler=pseudo_sampler, batch_size=params['batch_size'])
    model_name = 'pseudo_model.pth'

    f1_best = 0
    best_loss = float('inf')
    print(best_loss)

    for i in range(n_iter):
        print(f'Iter: {i+1}/{n_iter}')
        #print(len(pseudo_loader))
        model.eval()

        pseudo_labeled_X = []
        pseudo_labeled_y = []

        indices_to_remove = []
        correct_labels = 0
        all_labels = 0
        local_loss = float('inf')

        for batch_idx, (X, y) in enumerate(pseudo_loader):
            X = X.to(params['device'])
            y_pred = model(X).to(params['device'])
            probs = F.softmax(y_pred, dim = 1)
            conf, preds = torch.max(probs, 1) #процент уверенности, класс
            #print(conf, preds)
            for j in range(len(conf)):
                if conf[j] > threshhold:
                    global_index = batch_idx * pseudo_loader.batch_size + j
                    indices_to_remove.append(global_index)
                    pseudo_labeled_X.append(X[j].cpu())
                    pseudo_labeled_y.append(preds[j].cpu())
                    if preds[j] == y[j]:
                        correct_labels+=1
                    all_labels+=1
                    
        print(f'Длина новых данных после псведоразметки: {len(pseudo_labeled_y)} / {len(pseudo_data)}, {len(pseudo_labeled_y) / len(pseudo_data) * 100:.2f}')
        print(f'Точность псведомаркировки: {correct_labels}/{all_labels}, процент:{correct_labels / all_labels * 100}')
        pseudo_labeled_X = torch.stack(pseudo_labeled_X)
        pseudo_labeled_y = torch.stack(pseudo_labeled_y)

        after_data = TensorDataset(pseudo_labeled_X, pseudo_labeled_y)

        labels = [label for _, label in after_data]
        class_counts = np.bincount(labels)

        print(class_counts)
        combined_data = torch.utils.data.ConcatDataset([train_data, after_data])
        combined_sampler = solve_imbalance(combined_data)
        combined_loader = DataLoader(combined_data, sampler = combined_sampler, batch_size=params['batch_size'])

        new_indices = [idx for idx, _ in enumerate(pseudo_data.indices) if idx not in indices_to_remove]
        pseudo_data.indices = [pseudo_data.indices[i] for i in new_indices]
        pseudo_loader = DataLoader(pseudo_data, shuffle=False, batch_size=pseudo_loader.batch_size)
        train_data = combined_data

        val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)
        
        if len(pseudo_loader) < 0:
            print('Обучение закончилось, т.к модель не нашла новые данные для обучения')
            break
        print('Начало обучения на псведо разметке')
        loss_fn = nn.CrossEntropyLoss()
            
        counter_pat = 0
        optim = torch.optim.Adam(params=model.parameters(), lr = params['lr'] * 0.33, weight_decay=0.01)

        model.train()
        all_loss = []
        for epoch in range(epochs_retrain):
            train_loss = []
            for X, y in combined_loader:
                X = X.to(params['device'])
                y = y.to(params['device'])
                y_pred = model(X).to(params['device'])
                optim.zero_grad()
                loss = loss_fn(y_pred, y)
                loss.backward()
                optim.step()
                train_loss.append(loss.cpu().detach().numpy())
            print(f'Epoch: {epoch+1}/{epochs_retrain}, Loss: {np.mean(train_loss):.5f}')
            all_loss.append(np.mean(train_loss))
        
            model.eval()
            test_loss = []
            
            patience = 2
            y_preds = []
            y_trues = []
            

            for X, y in val_loader:
                X = X.to(params['device'])
                y = y.to(params['device'])
                y_trues.extend(y.cpu().numpy())
                y_pred = model(X).to(params['device'])
                val_loss = loss_fn(y_pred, y)
                test_loss.append(val_loss.cpu().detach().numpy())
                y_pred = torch.argmax(y_pred, dim=1)
                y_preds.extend(y_pred.cpu().numpy())

            f1 = f1_score(y_trues, y_preds, average='weighted')
            print(f'F1: {f1:.6f}, Test Loss {np.mean(test_loss)}\n')

            #print(best_loss, np.mean(test_loss))
            if best_loss >= np.mean(test_loss):
                f1_best = f1
                best_loss = np.mean(test_loss)
                print('Была сохранена модель с наименьшим лоссом и метрикой:', best_loss)
                torch.save(model.state_dict(), model_name) 

            if local_loss >= np.mean(test_loss):
                local_loss = np.mean(test_loss)
                counter_pat = 0
            else:
                counter_pat+=1
                
            if counter_pat >= patience:
                print(f'Ранняя остановка на {1+epoch} эпохе')
                break

            
    print(f'Лучшая модель была с наименьшим лоссом: {best_loss}, её веса сохранены в файл: {model_name}')
    return f1_best



def train_newdata(train_data, val_data, new_data, params, clear_train = True):
    combined_data = torch.utils.data.ConcatDataset([train_data, new_data])
    epochs = 10
    model = funcs.match_case(params=params)
    if clear_train == True:
        model.new_class(params, epochs, train_data=combined_data, val_data=val_data, freeze_param=0)
    else:
        model.new_class(params, epochs, train_data=combined_data, val_data=val_data, freeze_param=1)

    return model