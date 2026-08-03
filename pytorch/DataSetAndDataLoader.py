import os

import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #Data loading

        xy = np.loadtxt(os.path.join('','./wine.csv'),delimiter=',', dtype=np.float32, skiprows=1) # remove first row as it is header
        self.x = torch.from_numpy(xy[:,1:]) #Don't first column
        self.y = torch.from_numpy(xy[:,[0]]) # n_samples, 1
        self.sample_num = self.x.shape[0]
        print(f'x ={self.x}, 'f'y ={self.y}, xy ={xy}, self.sample_num ={self.sample_num}')
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.sample_num

if __name__ == '__main__':
    dataset = WineDataset()
    first_dataset = dataset[0]
    print(f'first_dataset={first_dataset}')
    features,labels = first_dataset
    print(f'features={features}, labels={labels}')
    dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True, num_workers=2)
    dataloader_iter = iter(dataloader)
    data = next(dataloader_iter)
    features,labels = data
    print(f'features={features}, labels={labels}')

    num_epochs = 100
    total_samples = len(dataset)
    num_iteration = math.ceil(total_samples / 4) # total_samples/batch_size
    print(f'num_iteration={num_iteration}, total_samples={total_samples}')

    for epoch in range(num_epochs):
        for i, (input,label) in enumerate(dataloader):
            # forward backward, update
            if (i+1) % 5 == 0:
                print(f'epoch={epoch + 1}, input={input}, label={label}, step={i+1}')