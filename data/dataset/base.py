from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .util import preprocessing


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

    def __getitem__(self, idx):
        assert self.is_preprocess
        return [x[idx] for x in self.xs], self.labels[idx]

    def __len__(self):
        return len(self.labels)
