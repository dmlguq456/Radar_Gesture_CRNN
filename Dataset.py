import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob

class RadarGesture(Dataset):
    def __init__(self, root_dir):
        entry = []
        files = glob.glob1(root_dir,'*.csv')
        for f in files:
            f = os.path.join(root_dir,f)
            entry.append(f)
        self.entry = sorted(entry)

    def __len__(self):
        return len(self.entry)

    def __getitem__(self, idx):
        single_csv_path = self.entry[idx]
        single_csv = np.genfromtxt(single_csv_path, delimiter=',')
        single_np = single_csv.reshape(single_csv.shape[0],-1,20)
        if single_np.shape[1] < 80:
            print("It doesn't have frame length of 80. Its length is " + str(single_np.shape[1]))
            dif = 80 - single_np.shape[1]
            single_np = np.append(single_np,np.zeros((6,dif,20)),axis=1)   
        single_tensor = torch.from_numpy(single_np)
        single_tensor = torch.DoubleTensor(single_tensor).float()
        single_label = int(single_csv_path[-5])

        return (single_tensor, single_label)

if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'
    train_dataset = RadarGesture(train_dir)
    test_dataset = RadarGesture(test_dir)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

    it = iter(train_loader)
    for i in range(50):
        image, label = next(it)
        print(label)
    print(len(train_loader))

    image, label = next(iter(test_loader))
    for x,y in test_loader:
        print(x.shape,y)
    print(len(test_loader))

    image, label = next(iter(train_loader))
    for x,y in train_loader:
        print(x.shape,y)
    print(len(train_loader))