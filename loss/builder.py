import torch.nn as nn
import copy
from .celoss import CELoss

class Losses(nn.Module):
    def __init__(self, loss_cfg):
        super(Losses, self).__init__()
        self.loss_cfg = loss_cfg

    def forward(self, 
                zs, 
                qs, 
                qf,
                label):
        losses = {}
        _loss = 0
        for name, cfg in self.loss_cfg.items():
            config = copy.deepcopy(cfg)
            weight = config.pop("weight")
            loss = eval(name)(**config)(zs=zs, 
                                        qs=qs, 
                                        qf=qf,
                                        label=label)
            losses.update({name: weight*loss})
            _loss += (weight*loss)
        losses.update({"loss": _loss})
        return losses

def build(loss_cfg,
          zs,
          qs,
          qf,
          label):
    losses = Losses(loss_cfg)
    loss = losses(zs, qs, qf, label)
    return loss