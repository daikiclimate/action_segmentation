import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import label_to_id


def build_loss_func(config: dict, weight: list = None):
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = Ce_Bce_Loss(config.pairs, weight)
    criterion = FocalLossWithOutOneHot(gamma=4)

    return criterion


class Ce_Bce_Loss(nn.Module):
    def __init__(self, pairs: list, weight: list):
        super().__init__()
        self._pairs = pairs
        self._ce = nn.CrossEntropyLoss(weight=weight)
        self._bce_loss = nn.BCELoss()
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._bce_loss = nn.MSELoss()
        self._softmax = nn.Softmax(dim=1)
        self._bce_weight = 0.5

    def forward(self, outputs, labels):
        total_loss = self._ce(outputs, labels)
        outputs = self._softmax(outputs)

        for pair in self._pairs:
            label1, label2 = [torch.tensor(label_to_id(i)) for i in pair]
            index = labels == label1
            wrong_out = outputs[index, label2]
            if len(wrong_out) > 0:
                t = wrong_out.shape
                zeros_label = torch.zeros(len(wrong_out)).float()
                loss = self._bce_loss(wrong_out, zeros_label)
                loss *= self._bce_weight
                total_loss += loss
            label1, label2 = label2, label1
            index = labels == label1
            wrong_out = outputs[index, label2]
            if len(wrong_out) > 0:
                t = wrong_out.shape
                zeros_label = torch.zeros(len(wrong_out)).float()
                loss = self._bce_loss(wrong_out, zeros_label)
                loss *= self._bce_weight
                total_loss += loss
        return total_loss


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = (
            loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma
        )  # focal loss
        return loss.sum()
