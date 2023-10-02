import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, 
                qs, 
                qf,
                label,
                **kwargs):
        loss = 0
        for q in qs:
            loss += self.celoss(q, label)
        loss += self.celoss(qf, label)
        return loss