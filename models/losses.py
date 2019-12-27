import torch
import torch.nn as nn
import numpy as np
# currently not used for this project

class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1).mean() / 2.
        return loss