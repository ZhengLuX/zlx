import torch
import torch.optim as optim
import math

class CosineAnnealingWarmupRestarts(optim.lr_scheduler.LRScheduler):
    """
    Cosine Annealing Warmup Restarts学习率调度器。

    参数:
    - optimizer (Optimizer): 包装的优化器。
    - first_cycle_steps (int): 第一个周期的步数。
    - cycle_mult (float): 周期步数的乘数。默认: 1.0。
    - max_lr (float): 第一个周期的最大学习率。默认: 0.1。
    - min_lr (float): 最小学习率。默认: 0.001。
    - warmup_steps (int): 线性warmup的步数。默认: 0。
    - gamma (float): 周期间最大学习率的减少率。默认: 1.0。
    - last_epoch (int): 上一个周期的索引。默认: -1。
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1):
        # 确保warmup步骤少于第一个周期的步骤
        assert warmup_steps < first_cycle_steps

        # 初始化各种参数
        self.first_cycle_steps = first_cycle_steps  # 第一个周期的步数
        self.cycle_mult = cycle_mult  # 周期步数的乘数
        self.base_max_lr = max_lr  # 第一个周期的最大学习率
        self.max_lr = max_lr  # 当前周期的最大学习率
        self.min_lr = min_lr  # 最小学习率
        self.warmup_steps = warmup_steps  # 线性warmup的步数
        self.gamma = gamma  # 周期间最大学习率的减少率

        # 当前周期的步数和周期计数
        self.cur_cycle_steps = first_cycle_steps  # 当前周期的步数
        self.cycle = 0  # 周期计数
        self.step_in_cycle = last_epoch  # 当前周期的步数

        # 调用父类构造函数
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # 初始化学习率
        self.init_lr()

    def init_lr(self):
        """
        初始化每个参数组的学习率。
        """
        self.base_lrs = []  # 存储每个参数组的基础学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr  # 将学习率初始化为最小学习率
            self.base_lrs.append(self.min_lr)  # 添加到基础学习率列表

    def get_lr(self):
        """
        计算当前步骤的学习率。

        返回:
        - list: 当前步骤的学习率列表。
        """
        if self.step_in_cycle == -1:
            return self.base_lrs  # 如果没有进行任何步骤，返回基础学习率
        elif self.step_in_cycle < self.warmup_steps:
            # 线性warmup阶段
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            # Cosine Annealing阶段
            return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(
                math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        执行一步学习率调度。

        参数:
        - epoch (int, 可选): 当前周期。如果为None，则使用内部的last_epoch值并自增。
        """
        if epoch is None:
            # 自增步骤并检查是否需要重置周期
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1  # 当前周期的步数自增

            # 检查当前周期的步数是否超过了周期限制
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1  # 周期计数自增
                self.step_in_cycle -= self.cur_cycle_steps  # 重置当前周期的步数
                # 计算下一个周期的步数
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            # 根据给定的epoch更新周期和步骤
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        # 更新最大学习率
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)  # 更新上一个周期的索引

        # 更新优化器中的学习率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr  # 设置当前学习率