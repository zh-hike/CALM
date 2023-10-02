import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, qf, label, **kwargs):
        pred = qf.argmax(dim=-1)
        n = qf.shape[0]
        breakpoint()
        return (pred == label).sum() / n