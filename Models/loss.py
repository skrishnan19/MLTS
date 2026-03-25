import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from Models.Util import *
import torch.nn.functional as F
from torch import Tensor

class CELoss(nn.Module):
    def __init__(self, unique_lbls, gpuid):
        super(CELoss, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.unique_lbls = unique_lbls
        self.gpuid = gpuid

    def forward(self, logits, targets, cw=None, mask = None):
        loss = F.cross_entropy(logits, targets, reduction='none').view(-1)
        if mask is None:
            mask = torch.ones_like(loss).float()
        if cw is None:
            cw = torch.ones_like(self.unique_lbls).float()
        cw = cw[targets]

        assert cw.shape == mask.shape
        cw = cw * mask

        assert loss.shape == cw.shape
        loss = loss * cw
        loss = loss.sum() / cw.sum()
        return loss


class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()

    def D(self, p, z):
        sim = F.cosine_similarity(p, z.detach(), dim=-1)
        return 2 - 2*sim #.mean()

    def forward(self, fea_s, fea_t):
        return self.D(fea_s, fea_t)



class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()

    def D(self, p, z):
        sim = F.cosine_similarity(p, z.detach(), dim=-1)
        return 2 - 2*sim #.mean()

    def forward(self, fea_s, fea_t):
        return self.D(fea_s, fea_t)




