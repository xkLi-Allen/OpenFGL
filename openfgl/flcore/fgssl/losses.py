from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F



def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()
class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss
class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()



def bt_loss(h1: torch.Tensor, h2: torch.Tensor, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    if lambda_ is None:
        lambda_ = 1. / feature_dim

    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = h1.T @ h2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss
class BarlowTwins(Loss):
    def __init__(self, lambda_: float = None, batch_norm: bool = True, eps: float = 1e-5):
        self.lambda_ = lambda_
        self.batch_norm = batch_norm
        self.eps = eps

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        loss = bt_loss(anchor, sample, self.lambda_, self.batch_norm, self.eps)
        return loss.mean()