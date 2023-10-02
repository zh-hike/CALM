import torch
import torch.nn as nn


class SingleViewBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleViewBackbone, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.backbone(x)


class MultiViewBackbone(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MultiViewBackbone, self).__init__()
        nets = []
        for input_dim, hidden_dim, output_dim in zip(
            input_dims, hidden_dims, output_dims
        ):
            nets.append(SingleViewBackbone(input_dim, hidden_dim, output_dim))
        self.backbones = nn.ModuleList(nets)

    def forward(self, xs):
        res = []
        for x, backbone in zip(xs, self.backbones):
            res.append(backbone(x))
        return res
