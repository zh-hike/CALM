from .base import BaseRunner
from arch.builder import build as arch_build
from data.builder import build as data_build
from loss.builder import build as loss_build
from metric.builder import build as metric_build
from functools import partial
from torch.optim import Adam
import torch

class Runner(BaseRunner):
    def __init__(self, 
                 arch_cfg=None,
                 data_cfg=None,
                 global_cfg=None,
                 loss_cfg=None,
                 metric_cfg=None):
        
        self.arch_cfg = arch_cfg
        self.data_cfg = data_cfg
        self.loss_cfg = loss_cfg
        self.metric_cfg = metric_cfg
        super(Runner, self).__init__(global_cfg['device'], 
                                     global_cfg['epochs'])
        

    def init_net(self):
        self.net = arch_build(**self.arch_cfg)
        self.net.to(self.device)

    def init_metric_func(self):
        self.metric_func = partial(metric_build, metric_cfg=self.metric_cfg)

    def init_loss_func(self):
        self.loss_func = partial(loss_build, loss_cfg=self.loss_cfg)

    def init_optimizer(self):
        self.opt = Adam(self.net.parameters(),
                        lr=1e-3,
                        weight_decay=1e-5)
        
    def init_dataset(self):
        self.train_dl, self.val_dl = data_build(**self.data_cfg['Dataset'],
                                                batch_size=self.data_cfg['DataLoader']['batch_size'])
        
    def train_an_batch(self):
        zs, qs, qf = self.net(self.inputs)
        self.loss = self.loss_func(zs=zs, 
                                   qs=qs, 
                                   qf=qf, 
                                   label=self.target)
        self.metric = self.metric_func(qs=qs,
                                       qf=qf,
                                       label=self.target)
        self.opt.zero_grad()
        self.loss['loss'].backward()
        self.opt.step()

    @torch.no_grad()
    def val_an_batch(self):
        zs, qs, qf = self.net(self.inputs)
        self.loss = self.loss_func(zs=zs, 
                                   qs=qs, 
                                   qf=qf, 
                                   label=self.target)
        
        self.metric = self.metric_func(qs=qs,
                                       qf=qf,
                                       label=self.target)
