import torch
import torch.nn as nn

from arch.base import MultiViewBackbone


class ClassifierHead(nn.Module):
    def __init__(self, hidden_dims, num_class):
        res = []
        for hidden_dim in hidden_dims:
            _classifier = nn.Sequential(nn.Linear(hidden_dim, num_class))
            res.append(_classifier)
        self.classifier_head = nn.ModuleList(res)

    def forward(self, zs):
        res = []
        for z, classifier in zip(zs, self.classifier_head):
            res.append(classifier(z))
        return res


class CALM(nn.Module):
    def __init__(
        self, input_dims, hidden_dims, output_dims, num_class, fusion_method="CaF"
    ):
        self.backbones = MultiViewBackbone(input_dims, hidden_dims, output_dims)
        self.num_view = len(input_dims)
        self.classifier = ClassifierHead(hidden_dims, num_class)
        self.fusion_method = fusion_method

    def forward(self, xs):
        zs = self.backbones(xs)
        breakpoint()
        qs = self.classifier(zs)

        return qs
