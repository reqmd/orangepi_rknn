import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LabeledDataset(Dataset):
    def __init__(self, root, transform = None, test=False):
        self.root = root
        self.classes = os.listdir(root)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transfrom = transform
        self.images = []
        self.labels = []
        self.root_images = []
        self.image_name = ''
        for class_name in self.classes:
            self.class_root = os.path.join(self.root, class_name)
            for image_name in os.listdir(self.class_root):
                self.images.append(image_name)
                self.labels.append(self.class_to_idx[class_name])
                if test == True:
                    self.root_images.append(os.path.join(self.class_root, image_name))
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(os.path.join(self.root, self.classes[label], self.images[idx])).convert('RGB')
        self.image_name = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype = torch.int64)

        if self.transfrom != None:
            image = self.transfrom(image)
        return image, label
    
class TrainTestSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    