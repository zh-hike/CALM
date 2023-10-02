import scipy.io as io
from sklearn.model_selection import train_test_split

from .base import BaseDataset


class HandWritten(BaseDataset):
    def __init__(self, 
                 data_root, 
                 test_size=0.2, 
                 random_state=1, 
                 train=True):
        data = io.loadmat(data_root)
        xs = data["X"][0]
        labels = data["Y"].squeeze().astype("int64")

        xs = self.preprocess(xs)
        self.xs, self.labels = self.split_train_test(xs,
                                                     labels,
                                                     test_size=test_size,
                                                     random_state=random_state,
                                                     train=train)