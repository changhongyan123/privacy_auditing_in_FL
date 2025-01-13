# data laoder
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np


class MedicalMNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir + "/../medicalmnist/all_data.csv")
        self.root_dir = root_dir + "/../medicalmnist/archive/"
        self.transform = transform
        self.data, self.target = self.__build_truncated_dataset__()

    def __len__(self):
        return len(self.annotations)

    def __build_truncated_dataset__(self):
        data = []
        target = []
        for i in range(len(self.annotations)):
            img_path = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            image = io.imread(img_path)
            y_label = torch.tensor(int(self.annotations.iloc[i, 1]))
            data.append(image)
            target.append(y_label)
        data = np.array(data)
        target = np.array(target)
        return data, target

    def __getitem__(self, index):
        image, y_label = self.data[index], self.target[index]
        if self.transform:
            image = self.transform(image)
        
        return image, y_label
