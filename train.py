import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score

from dataset import LabeledDataset

def train_objective(
    train_data,
    train_loader, 
    valid_loader,  
    model, 
    optim, 
    loss_fn, 
    device,
    epochs,
    pseudo_labeling = None
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
        
        for X, y in valid_loader:
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

    if pseudo_labeling != None:
        pseudo_label(
            model = model,
            pseudo_root = pseudo_labeling,
            train_data = train_data,
            valid_loader = valid_loader,
            threshhold=0.95,
            n_iter = 3,
            epochs_retrain=5,
            batch_size= 8,
            stratify = None
        )
    return total_loss[-1]



def train_final(
    train_data,
    train_loader, 
    valid_loader,  
    model, 
    optim, 
    loss_fn, 
    device,
    epochs,
    f1_threshhold,
    model_name = 'best_model.pth',
    batch_size = None,
    pseudo_labeling = None
):
    
    total_loss = []
    best_loss = float('inf')
    counter = 0
    f1_best = 0
    f1_best_epoch = 0

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
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss{np.mean(train_loss)}')
    
        model.eval()
        test_loss = []
        y_preds = []
        y_trues = []
        patience = 3
        
        for X, y in valid_loader:
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
        
        if pseudo_labeling != None:
            pseudo_label(
                model = model,
                pseudo_root = pseudo_labeling,
                train_data = train_data,
                optim=optim,
                threshhold=0.95,
                n_iter = 3,
                epochs_retrain=5,
                batch_size= batch_size,
                stratify = None
            )
    print(f'Лучшая метрика была достигнута на {f1_best_epoch} эпохе, значение f1 {f1_best:.6f}')
    return f1_best



def pseudo_label(model, 
                 pseudo_root, 
                 train_data,   
                 device,
                 optim,
                 threshhold = 0.95,
                 n_iter = 3,
                 epochs_retrain = 5, 
                 batch_size = 8, 
                 stratify = None
                ):
    print('Начало псведо разметки')
    val_transform = transforms.Compose([
        transforms.Resize((train_data[0][0].shape[1], train_data[0][0].shape[2])), 
        transforms.ToTensor()
    ])
    full_data = LabeledDataset(pseudo_root, transform=val_transform)
    random_indices = np.random.permutation(len(full_data))
    pseudo_data = Subset(full_data, random_indices[:1000])
    pseudo_loader = DataLoader(pseudo_data, shuffle=True, batch_size=batch_size)
    model_name = 'best_model.pth'
    for i in range(n_iter):
        print(f'Iter: {i+1}/{n_iter}')
        #print(len(pseudo_loader))
        model.eval()
        pseudo_labeled_X = []
        pseudo_labeled_y = []
        indices_to_remove = []
        correct_labels = 0
        all_labels = 0
        for batch_idx, (X, y) in enumerate(pseudo_loader):
            X = X.to(device)
            y_pred = model(X).to(device)
            #X_view = X.view(-1, 3*32*32).to(device)
            #y_pred = model(X_view).to(device)
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
        #print(pseudo_labeled_X, pseudo_labeled_y)
        after_data = TensorDataset(pseudo_labeled_X, pseudo_labeled_y)
        #combined_loader = DataLoader(after_data, shuffle=True, batch_size=batch_size)
        combined_data = torch.utils.data.ConcatDataset([train_data, after_data])
        if len(combined_data) > 200 and len(combined_data) < 500:
            batch_size*=2
        combined_loader = DataLoader(combined_data, shuffle=True, batch_size=batch_size)
        new_indices = [idx for idx, _ in enumerate(pseudo_data.indices) if idx not in indices_to_remove]
        pseudo_data.indices = [pseudo_data.indices[i] for i in new_indices]
        pseudo_loader = DataLoader(pseudo_data, shuffle=False, batch_size=pseudo_loader.batch_size)
        train_data = combined_data
        
        if len(pseudo_loader) < 0:
            print('Обучение закончилось, т.к модель не нашла новые данные для обучения')
            break
            #model = SimpleModel(num_classes=5).to(device)
        print('Начало обучения на псведо разметке')
        if stratify != None:
            loss_fn = nn.CrossEntropyLoss(weight=stratify)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        counter_pat = 0
        model.train()
        for epoch in range(epochs_retrain):
            best_loss = float('inf')
            train_loss = []
            for X, y in combined_loader:
                #print(X, y)
                #X = X.view(-1, 3*32*32).to(device)
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                optim.zero_grad()
                loss = loss_fn(y_pred, y)
                loss.backward()
                optim.step()
                train_loss.append(loss.cpu().detach().numpy())
            print(f'Epoch: {epoch+1}/{epochs_retrain}, Loss: {np.mean(train_loss):.5f}')
        
            #model.eval()
            #test_acc = []
            #test_loss = []

            #best_loss = float('inf')
            
            #patience = 2
            #for X, y in valid_loader:
                #X = X.view(-1, 3*32*32).to(device)
                #X = X.to(device)
                #y = y.to(device)
                #y_pred = model(X)
                ##val_loss = loss_fn(y_pred, y)
                #test_loss.append(val_loss.cpu().detach().numpy())
                #early_stop(val_loss, model)
                #y_pred = torch.argmax(y_pred, dim=1)
                #acc = sum(y == y_pred) / len(y)
                #test_acc.append(acc.cpu().detach().numpy())
            #print(f'Accuracy: {np.mean(test_acc) * 100:.2f}, Test Loss: {np.mean(test_loss)}\n')

            #if best_loss > np.mean(test_loss):
                #best_loss = np.mean(test_loss)
                #counter_pat = 0
                #torch.save(model.state_dict(), model_name) 
            #else:
                #counter_pat+=1
                
            #if counter_pat >= patience:
                #break
            #print(counter_pat)
            #print(batch_size)
    print(f'Лучшая модель была с точностью: {best_loss}, её веса сохранены в файл: {model_name}')
    return model