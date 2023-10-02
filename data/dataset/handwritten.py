import scipy.io as io
from sklearn.model_selection import train_test_split

from .base import BaseDataset


class HandWritten(BaseDataset):
    def __init__(self, data_root, test_size=0.2, random_state=1, train=True):
        data = io.loadmat(data_root)
        xs = data["X"][0]
        labels = data["Y"].squeeze()

        xs = self.preprocess(xs)
        train_xs, val_xs, train_y, val_y = train_test_split(
            xs, labels, test_size=test_size, random_state=random_state
        )
        self.xs = []
        self.labels = []
        if train:
            self.xs = train_xs
            self.labels = train_y
        else:
            self.xs = val_xs
            self.labels = val_y
