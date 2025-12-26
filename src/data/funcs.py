from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
import shutil
import os

from .transforms import return_transforms
from .dataset import TrainTestSubset, LabeledDataset
from sklearn.model_selection import train_test_split as tts

def solve_imbalance(data):
    labels = [label for _, label in data]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=int(np.min(class_counts) * len(class_counts)), replacement=False)
    return sampler

def train_test_split(data, resolution, test_size=0.5):
    indices = list(range(len(data)))
    train_indices, val_indices = tts(indices, test_size=test_size, random_state=42, stratify=data.labels)
    train_transform, val_transform = return_transforms(resolutions=resolution)
    train_data = TrainTestSubset(data, train_indices, train_transform)
    val_data = TrainTestSubset(data, val_indices, val_transform)
    return train_data, val_data

def dataset_into_loader(data, batch_size):
    if not isinstance(data, list):
        sampler = solve_imbalance(data)
        loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return loader
    
    else:
        train_data, val_data = data

        train_sampler = solve_imbalance(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_sampler = solve_imbalance(val_data)
        val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        return train_loader, val_loader
    
def create_annot(data, txt_root):
    with open(txt_root, 'w') as f: 
        for st in data.root_images:
            f.write(f'{st}\n')

def replace_new_ftp_data(src_root, dst_root):
    all_files = os.listdir(src_root)
    c = 0
    for f in all_files:
        _, ext = os.path.splitext(f)
        if ext == '.bmp':
             src_path = os.path.join(src_root, f)
             dst_path = os.path.join(dst_root, f)
             shutil.move(src_path, dst_path)
             c+=1
    if c == 0:
        print('ФАЙЛЫ НЕ БЫЛИ ПОЛУЧЕНЫ ИЗ ПАПКИ FTP')
        return 1
    print(f'Успешно перемещено {c} файлов')

