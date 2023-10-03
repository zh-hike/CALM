import torch
import torch.nn as nn

from arch.base import MultiViewBackbone
from arch.CaF import MultiClassCaE
from torch.nn import functional as F

class ClassifierHead(nn.Module):
    def __init__(self, hidden_dims, num_class):
        super(ClassifierHead, self).__init__()
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
        self, 
        input_dims, 
        hidden_dims, 
        output_dims, 
        num_class, 
        fusion_method="CaF",
        tau=0.3,
    ):
        super(CALM, self).__init__()
        self.backbones = MultiViewBackbone(input_dims, hidden_dims, output_dims)
        self.num_view = len(input_dims)
        self.classifier = ClassifierHead(output_dims, num_class)
        self.fusion_method = fusion_method

        self.cae = MultiClassCaE(num_class)
        self.tau = tau

    def forward(self, xs, label=None):
        zs = self.backbones(xs)
        qs = self.classifier(zs)     # 每个视图的分类概率
        cae_loss = 0
        if self.training:
            cae_loss = self.cae([q.detach() for q in qs], label)
        else:
            breakpoint()
            label = self.recompute_label(qs)

        u_v = self.cae.pdf([q.detach() for q in qs], label)
        q_f = self.fusion(qs, u_v)         # 融合后的概率
        u_f = self.cae.pdf([q_f], label)
        
        return zs, qs, q_f, cae_loss, u_f
    
    def fusion(self, qs, u_v=None):
        if self.fusion_method == "equal":
            return sum(qs) / len(qs), None
        elif self.fusion_method == "caf":
            breakpoint()
            qs = torch.stack(qs)
            uuv = F.softmax(u_v / self.tau, dim=0).unsqueeze(-1)
            qf = (uuv * qs).sum(dim=0)
            return qf

    def recompute_label(self, qs):
        qs = torch.stack(qs)
        qs = F.softmax(qs, dim=-1)
        qs = qs.prod(dim=0)
        return qs.argmax(dim=1)

