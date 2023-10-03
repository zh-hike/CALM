import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import numpy as np
Eps = 1e-4


class SingleClassCaE(nn.Module):
    """
    signle class Confidence-aware Evaluator
    """

    def __init__(self, dim):
        super(SingleClassCaE, self).__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, requires_grad=True))
        self.sigma = nn.Parameter(torch.ones(dim, requires_grad=True))

    def _reset(self):
        self.sigma.clip_(Eps, 10000)

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(1, -1)
        sigma2 = self.sigma**2
        diff = x - self.mu
        sigma_ = 1 / sigma2
        item1 = (diff * sigma_ * diff).sum(dim=1).mean()
        item2 = torch.log(sigma2.prod())
        return (item1 + item2) / 2

    def pdf(self, x):
        # breakpoint()
        sigma2 = (self.sigma * self.dim) ** 2
        _mu = self.mu.detach().cpu()
        _sigma = torch.eye(self.dim) * sigma2.detach().cpu()
        score = multivariate_normal.pdf(x, _mu, _sigma) / multivariate_normal.pdf(
            _mu, _mu, _sigma
        )
        if isinstance(score, (int, float)):
            score = np.array([score])
        return torch.from_numpy(score).to(x.device)


class MultiClassCaE(nn.Module):
    def __init__(self, dim):
        super(MultiClassCaE, self).__init__()
        self.cae = nn.ModuleList([SingleClassCaE(dim) for _ in range(dim)])

    def forward(self, preds, pseudo_label):
        assert isinstance(preds, list)
        assert preds[0].dim() == 2
        assert pseudo_label.dim() == 1
        preds = torch.stack(preds)
        losses = []
        for i in range(preds.shape[1]):
            u = self.cae[pseudo_label[i]](preds[:, i, ...])
            losses.append(u)
        loss = sum(losses) / len(losses)
        return loss

    def _reset(self):
        for net in self.cae:
            net._reset()

    def pdf(self, preds, label):
        device = label.device
        label = label.cpu()
        preds = torch.stack(preds)
        preds = preds.detach().cpu()
        scores = []
        for i in range(preds.shape[1]):
            u = self.cae[label[i]].pdf(preds[:, i, ...])
            scores.append(u.reshape(-1, 1))
        scores = torch.concat(scores, dim=-1)
        scores = scores.to(device)
        return scores
