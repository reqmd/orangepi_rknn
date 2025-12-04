import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, ConcatDataset
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



def calculate_avg_size_per_class(dataset: Dataset, num_classes: int):
    class_sizes = {i: [] for i in range(num_classes)}

    for image, label in dataset:
        h, w = image.shape[-2:]  
        class_sizes[label].append((h, w))

    avg_sizes = {}
    max_avg_height = 0
    max_avg_width = 0

    for cls in class_sizes:
        sizes = np.array(class_sizes[cls])
        heights, widths = sizes[:, 0], sizes[:, 1]

        def filter_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]

        filtered_heights = filter_outliers(heights)
        filtered_widths = filter_outliers(widths)

        avg_h = np.mean(filtered_heights)
        avg_w = np.mean(filtered_widths)

        avg_sizes[cls] = (avg_h, avg_w)

        if avg_h > max_avg_height:
            max_avg_height = avg_h
        if avg_w > max_avg_width:
            max_avg_width = avg_w

    return avg_sizes, (max_avg_height, max_avg_width)

def full_loop_of_training(
    train_data,
    train_loader, 
    valid_loader, 
    test_dataset, 
    model, 
    optim, 
    loss_fn, 
    device,
    epochs,
    model_name,
    pseudo_labeling = None
):
    
    total_loss = []
    total_acc = []
    best_loss = float('inf')
    counter = 0

    print('Начат цикл обучения:')
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            #X = X.view(-1, 3*32*32).to(device)
            y = y.to(device)
            y_pred = model(X)
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            #scheduler.step()
            train_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {np.mean(train_loss):.5f}')
    
        model.eval()
        test_acc = []
        test_loss = []
        y_preds = []
        y_trues = []
        patience = 2
        
        for X, y in valid_loader:
            X = X.to(device)
            #X = X.view(-1, 3*32*32).to(device)
            y = y.to(device)
            y_trues.extend(y.cpu().numpy())
            y_pred = model(X)
            val_loss = loss_fn(y_pred, y)
            test_loss.append(val_loss.cpu().detach().numpy())
            #early_stop(val_loss, model)
            y_pred = torch.argmax(y_pred, dim=1)
            y_preds.extend(y_pred.cpu().numpy())
            #acc = sum(y == y_pred) / len(y)
            #test_acc.append(acc.cpu().detach().numpy())
        #print(f'Accuracy: {np.mean(test_acc) * 100:.2f}, Test Loss: {np.mean(test_loss)}\n')
        print(f'F1: {f1_score(y_trues, y_preds, average='weighted')}\n')
        
        total_loss.append(np.mean(test_loss))
        total_acc.append(np.mean(test_acc))
        if best_loss > np.mean(test_loss):
            best_loss = np.mean(test_loss)
            counter = 0
            torch.save(model.state_dict(), model_name) 
        else:
            counter+=1
        if counter >= patience:
            print('Ранняя остановка')
            break
    
    print(f'Лучшая модель была с наименьшим лосом: {best_loss}, её веса сохранены в файл: {model_name}')
    
    graph = int(input('Нужен ли вывод графиков: \n1.Да \nЛюбая другая клавиша: Нет'))
    if graph == 1:
        print('Вывод графиков:')
        plt.figure(figsize = (12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(total_loss, 'o', color = 'red')
        plt.title('Test Loss')
        plt.plot(total_loss)
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(total_acc, color = 'green')
        plt.title('Accuracy')
        plt.grid(True)
        
    confidence = int(input('Нужен ли вывод для классов: \n1.Да \nЛюбая другая клавиша: Нет'))
    if confidence == 1:
        print('Вывод уверенности классов после обучения:')
        np.set_printoptions(suppress=True)  
        eps = np.finfo(float).eps
        model.eval()
        for X, y in test_dataset:
            X = X.to(device)
            #X = X.view(-1, 3*32*32).to(device)
            y_pred = model(X.unsqueeze(0)).cpu().detach().numpy()
            e_x = np.exp(y_pred)
            s_x = e_x  / (np.sum(e_x , axis = -1, keepdims = True) + eps)
            print(y, f'{np.argmax(s_x, axis = 1)}, {np.round(s_x, 2)}')

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

def pseudo_label(model, 
                 pseudo_root, 
                 train_data, 
                 valid_loader, 
                 threshhold, 
                 n_iter, 
                 epochs_retrain = 3, 
                 batch_size = 8, 
                 stratify = None
                ):
    print('Начало псведо разметки')
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
        
            model.eval()
            test_acc = []
            test_loss = []

            best_loss = float('inf')
            
            patience = 2
            for X, y in valid_loader:
                #X = X.view(-1, 3*32*32).to(device)
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                val_loss = loss_fn(y_pred, y)
                test_loss.append(val_loss.cpu().detach().numpy())
                #early_stop(val_loss, model)
                y_pred = torch.argmax(y_pred, dim=1)
                acc = sum(y == y_pred) / len(y)
                test_acc.append(acc.cpu().detach().numpy())
            print(f'Accuracy: {np.mean(test_acc) * 100:.2f}, Test Loss: {np.mean(test_loss)}\n')

            if best_loss > np.mean(test_loss):
                best_loss = np.mean(test_loss)
                counter_pat = 0
                torch.save(model.state_dict(), model_name) 
            else:
                counter_pat+=1
                
            if counter_pat >= patience:
                break
            #print(counter_pat)
            #print(batch_size)
    print(f'Лучшая модель была с точностью: {best_loss}, её веса сохранены в файл: {model_name}')
    return model


labeled_root = 'Rice_Image_Dataset_Labeled'

labeled_dataset = LabeledDataset(labeled_root)
num_classes = len(labeled_dataset.classes)
avg_sizes, (max_h, max_w) = calculate_avg_size_per_class(labeled_dataset, num_classes)

print(max_h, max_w)

indices = list(range(len(labeled_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.4, random_state=42, stratify=labeled_dataset.labels)

train_transform = transforms.Compose([
    #transforms.Resize((224, 224)), #для MobileNet
    transforms.Resize((64, 64)), #для SimpleModel
    #transforms.Resize((32, 32)), #для MLP
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std),
    #transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    #transforms.ColorJitter(0.1, 0.1, 0.1)
])

val_transform = transforms.Compose([
    #transforms.Resize((224, 224)), #для MobileNet
    transforms.Resize((64, 64)), #для SimpleModel
    #transforms.Resize((32, 32)), #для MLP
    #transforms.Normalize(mean=mean, std=std),
    transforms.ToTensor()
])


train_dataset = TrainTestSubset(labeled_dataset, train_indices, transform=train_transform)
val_dataset = TrainTestSubset(labeled_dataset, val_indices, transform=val_transform)

batch_size = 8
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr=0.0001
#model = MobileNet(num_classes=5, alpha = 0.5).to(device)
model = SimpleModel(num_classes=5).to(device)
#model = MLP(input_size=3 * 32 * 32, hidden_size=512, output_size=5).to(device)
optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.01)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs = 25
model_name = 'best_model.pth'
test_root = 'rice_dataset/test'
test_dataset = LabeledDataset(test_root, transform=val_transform)

pseudo_root = 'rice_dataset/not_stratified_data'

import warnings

# Игнорировать все предупреждения
warnings.filterwarnings("ignore")

full_loop_of_training(
    train_data = train_data,
    train_loader = train_loader, 
    valid_loader = valid_loader, 
    test_dataset = test_dataset, 
    model = model, 
    epochs = epochs, 
    optim = optim, 
    loss_fn = loss_fn, 
    device = device,
    model_name = model_name,
    pseudo_labeling = None #pseudo_root
)

test_root = 'rice_dataset/test'
test_dataset = LabeledDataset(test_root, transform=val_transform)

np.set_printoptions(suppress=True)  
model = SimpleModel(num_classes=5).to(device)
model.load_state_dict(torch.load('best_model.pth'))
eps = np.finfo(float).eps
model.eval()
y_true = []
y_preds = []
for X, y in test_dataset:
    X = X.to(device)
    y_pred = model(X.unsqueeze(0)).cpu().detach().numpy()
    e_x = np.exp(y_pred)
    s_x = e_x  / (np.sum(e_x , axis = -1, keepdims = True) + eps)
    #print(y, f'{s_x}')
    #preds = np.argmax(s_x)
    #print(y, f'{preds}')
    y_true.append(y.cpu().detach().numpy())
    y_preds.append(np.argmax(s_x))
    
print(classification_report(y_true, y_preds))



