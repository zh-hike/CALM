import logging


class AverageMeter:
    def __init__(self, nd=4):
        self.sum = 0
        self.count = 0
        self.nd = nd

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.sum += (val * num)
        self.count += num

    def avg(self):
        return round(self.sum / self.count, self.nd)