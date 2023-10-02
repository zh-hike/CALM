from abc import ABC, abstractmethod

import torch


class BaseRunner(ABC):
    def __init__(self, devices, epochs):
        self.epochs = epochs

        self.init_device(devices)
        self.init_net()
        self.init_optimizer()
        self.init_dataset()

    def init_device(self, devices):
        if len(devices) == 1:
            device = self.devices[0]
            if device == -1:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")

    @abstractmethod
    def init_net(self):
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
            self.on_train_epoch_start()
            self.train_an_epoch(epoch_id)
            self.on_train_epoch_end()

    def train_an_epoch(self, epoch_id):
        for batch_id, (inputs, target) in enumerate(self.train_dl, 1):
            inputs = [i.to(self.device) for i in inputs]
            target = target.to(self.device)
            self.inputs = inputs
            self.target = target
            self.on_train_batch_start()
            self.train_an_batch()
            self.on_train_batch_end()

    @abstractmethod
    def train_an_batch(self):
        pass

    def on_train_start(self):
        self.net.train()

    def on_train_end(self):
        pass

    def on_validate_start(self):
        self.net.eval()

    def on_train_epoch_start(self):
        self.net.train()

    def on_train_epoch_end(self):
        pass

    def on_train_batch_start(self):
        pass

    def on_train_batch_end(self):
        pass
