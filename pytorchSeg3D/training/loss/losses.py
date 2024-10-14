import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from utils.ddp import AllGatherGrad


class SoftDiceLoss(nn.Module):

    def __init__(self, activation: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        初始化SoftDiceLoss类
        :param activation: 可选的激活函数（如softmax），用于将模型输出转换为概率
        :param batch_dice: 是否在批次级别计算Dice系数，默认为False
        :param do_bg: 是否计算背景类的Dice系数，默认为True
        :param smooth: 平滑因子，防止除零错误，默认为1.0
        :param ddp: 是否使用分布式数据并行计算，默认为True
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.activation = activation
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        """
        前向传播方法
         :param x: 模型的输出，通常是logits
         :param y: 真实标签
         :param loss_mask: 可选的损失掩码，用于指定有效区域
         :return: 计算得到的Dice损失
        """
        if self.activation is not None:
            x = self.activation(x)
        # 确定需要进行求和的轴，通常是空间维度
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            # 如果模型输出和真实标签的维度不一致，则调整真实标签的形状
            if x.ndim != y.ndim:
                #将y变形为（b, c(设为1), x, y(,z)）
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:

                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        score = 1 - dc
        return score

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        前向传播方法
        :param input: 模型的输出logits,不经过softmax层（不能是概率）
        :param target: 真实标签
        :return: 计算得到的交叉熵损失
        """
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())








class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        mask[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, mask):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        mask[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        mask[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(
                    dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - mask[:, i]).pow(2) * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + mask[:, i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - mask[:, i]).pow(2) * (1 - mask[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + (1 - mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - mask[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * mask[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)