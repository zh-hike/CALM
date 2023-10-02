import torch.nn as nn
import copy
from .accuracy import Accuracy

class Metrics(nn.Module):
    def __init__(self, metric_cfg):
        super(Metrics, self).__init__()
        self.metric_cfg = metric_cfg

    def forward(self, 
                qs, 
                qf,
                label):
        metrics = {}
        for name, cfg in self.metric_cfg.items():
            config = copy.deepcopy(cfg)
            weight = config.pop("weight")
            metric = eval(name)(**config)(qs=qs, 
                                          qf=qf,
                                          label=label)
            metrics.update({name: metric})
        return metrics

def build(metric_cfg,
          qs,
          qf,
          label):
    metrics = Metrics(metric_cfg)
    metric = metrics(qs, qf, label)
    return metric