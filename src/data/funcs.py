from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

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

