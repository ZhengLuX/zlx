import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def compute_confMatrix(mask_true: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):

    if ignore_mask is None:
        use_mask = np.ones_like(mask_true, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_true & mask_pred) & use_mask)
    fp = np.sum(((~mask_true) & mask_pred) & use_mask)
    fn = np.sum((mask_true & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_true) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


class LossAverage(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):

    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


