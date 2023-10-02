from abc import ABC, abstractmethod

import torch
import logging
from utils import AverageMeter

class BaseRunner(ABC):
    def __init__(self, devices, epochs):
        self.epochs = epochs

        self.init_device(devices)
        self.init_net()
        self.init_optimizer()
        self.init_dataset()
        self.init_loss_func()
        self.init_metric_func()

    def init_device(self, devices):
        if len(devices) == 1:
            device = devices[0]
            if device == -1:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")

    @abstractmethod
    def init_net(self):
        pass

    @abstractmethod
    def init_loss_func(self):
        pass

    @abstractmethod
    def init_metric_func(self):
        pass

    @abstractmethod
    def init_optimizer(self):
        pass

    @abstractmethod
    def init_dataset(self):
        """
        初始化训练集和验证集，dataloader等等
        """
        pass

    def validate(self):
        pass

    def train(self):
        self.on_train_start()

        for epoch_id in range(1, self.epochs + 1):
            self.epoch_id = epoch_id
            self.on_train_epoch_start()
            self.train_an_epoch()
            self.on_train_epoch_end()

            self.on_val_epoch_start()
            self.val_an_epoch()
            self.on_val_epoch_end()

        self.on_train_end()

    def train_an_epoch(self):
        for batch_id in range(1, len(self.train_dl_iter)+1):
            inputs, target = next(self.train_dl_iter)
            self.batch_id = batch_id
            inputs = [i.to(self.device) for i in inputs]
            target = target.to(self.device)
            self.num_sample = target.shape[0]
            self.num_view = len(inputs)
            self.inputs = inputs
            self.target = target
            self.on_train_batch_start()
            self.train_an_batch()
            self.on_train_batch_end()

    def val_an_epoch(self):
        for _ in range(1, len(self.val_dl_iter) + 1):
            inputs, target = next(self.val_dl_iter)
            inputs = [i.to(self.device) for i in inputs]
            target = target.to(self.device)
            self.num_sample = target.shape[0]
            self.num_view = len(inputs)
            self.inputs = inputs
            self.target = target
            self.on_val_batch_start()
            self.val_an_batch()
            self.on_val_batch_end()

    def on_train_start(self):
        self.acc_avg = AverageMeter()
        self.auc_avg = AverageMeter()
        self.loss_avg = AverageMeter()


    def on_train_end(self):
        pass

    def on_validate_start(self):
        self.net.eval()

    def on_validate_end(self):
        pass

    @abstractmethod
    def train_an_batch(self):
        pass

    @abstractmethod
    def val_an_batch(self):
        pass

    def on_train_epoch_start(self):
        self.train_dl_iter = iter(self.train_dl)
        self.net.train()
        self.acc_avg.reset()
        self.auc_avg.reset()
        self.loss_avg.reset()

    def on_val_epoch_start(self):
        self.val_dl_iter = iter(self.val_dl)
        self.net.eval()
        self.acc_avg.reset()
        self.auc_avg.reset()
        self.loss_avg.reset()

    def on_train_epoch_end(self):
        print(f"Epoch: {self.epoch_id}/{self.epochs}  train_acc: {self.acc_avg.avg()}  ", end="")

    def on_val_epoch_end(self):
        print(f"val_acc: {self.acc_avg.avg()}")

    def on_train_batch_start(self):
        pass

    def on_val_batch_start(self):
        pass

    def on_train_batch_end(self):
        self.acc_avg.update(self.metric["Accuracy"].item(), self.num_sample)
        self.loss_avg.update(self.loss['loss'].item(), self.num_sample)

    def on_val_batch_end(self):
        self.acc_avg.update(self.metric["Accuracy"].item(), self.num_sample)
        self.loss_avg.update(self.loss['loss'].item(), self.num_sample)
