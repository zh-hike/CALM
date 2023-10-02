from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .util import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

class BaseDataset(Dataset):
    def __init__(self):
        self.train_xs = []
        self.train_labels = []
        self.val_xs = []
        self.val_labels = []
        self.is_preprocess = False

    def preprocess(self, xs):
        self.is_preprocess = True
        xs = [preprocessing(x) for x in xs]
        return xs
    
    def split_train_test(self, 
                         xs, 
                         labels, 
                         test_size,
                         random_state=1,
                         train=True):
        n = len(labels)
        indexs = np.arange(n)
        train_idx, val_idx = train_test_split(indexs, 
                                              test_size=test_size,
                                              random_state=random_state)
        idx = train_idx if train else val_idx
        return [x[idx] for x in xs], labels[idx]

    def __getitem__(self, idx):
        assert self.is_preprocess
        return [x[idx] for x in self.xs], self.labels[idx]

    def __len__(self):
        return len(self.labels)
