import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):

    def __init__(self, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, labels):
        """
		preds:softmax输出结果
		labels:真实值
		"""
        eps = 1e-7
        preds = F.softmax(preds, dim=-1)
        ce = -1 * torch.log(preds + eps) * labels
        floss = torch.pow((1 - preds), self.gamma) * ce
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

