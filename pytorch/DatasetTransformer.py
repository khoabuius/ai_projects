import os

import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        #Data loading

        xy = np.loadtxt(os.path.join('','./wine.csv'),delimiter=',', dtype=np.float32, skiprows=1) # remove first row as it is header
        self.x = xy[:,1:] #Don't first column
        self.y = xy[:,[0]] # n_samples, 1
        self.sample_num = self.x.shape[0]
        print(f'x ={self.x}, 'f'y ={self.y}, xy ={xy}, self.sample_num ={self.sample_num}')
        self.transform = transform
    def __getitem__(self,idx):
        sample = self.x[idx], self.y[idx]
        return self.transform(sample) if self.transform else sample
    def __len__(self):
        return self.sample_num


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        input, label = sample
        input *= self.factor
        return input, label
if __name__ == '__main__':
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features,labels = first_data
    print(f'Before features={features}, labels={labels}, typeof={type(features)}, typeof={type(labels)}')

    composed_transform = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
    dataset = WineDataset(transform=composed_transform)
    first_data = dataset[0]
    features, labels = first_data
    print(f'After compose transform, features={features}, labels={labels}, typeof={type(features)}, typeof={type(labels)}')
