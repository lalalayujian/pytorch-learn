"""
调整学习率方法的策略，根据epoch训练次数来调整
LambdaLR: new_lr = lambda * initial_lr，每个epoch都更新，lambda可由epoch和initial_lr得到，需要自定义
StepLR: new_lr = initial_lr * gamma^(epoch//step_size), 每过step_size个epoch，做一次更新
MultiStepLR: new_lr = initial_lr * gamma^(epoch在milestones的位置), 每次遇到milestones中的epoch，做一次更新
ExponentialLR: new_lr = initial_lr * gamma^epoch，每个epoch都更新
ReduceLROnPlateau: new_lr = lambda * old_lr，当给定metric停止优化时减小lr
都可设置last_epoch可从中断位置继续训练
"""


import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        pass


net = model()

initial_lr = 0.1
optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

scheduler_dict = {
    # 'lambda_lr': LambdaLR(optimizer, lr_lambda=lambda epoch:  0.15 * epoch),
    'step_lr': StepLR(optimizer, step_size=3, gamma=0.5),
    # 'multi_step_lr': MultiStepLR(optimizer, milestones=[3, 7], gamma=0.5),
    'exponential_lr': ExponentialLR(optimizer, gamma=0.5)
}

print("初始化的学习率：", optimizer.defaults['lr'])
# scheduler = scheduler_dict['lambda_lr']
# scheduler = scheduler_dict['step_lr']
# scheduler = scheduler_dict['multi_step_lr']

# for epoch in range(10):
#     optimizer.zero_grad()
#     optimizer.step()
#
#     print('第%d个epoch的学习率：%f'% (epoch, optimizer.param_groups[0]['lr']))
#     scheduler.step()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
# 当指标不再减小时，2个epoch之后再减小学习率，倍率为0.5
for epoch in range(10):
    loss = 2
    optimizer.zero_grad()
    optimizer.step()

    print('第%d个epoch的学习率：%f'% (epoch, optimizer.param_groups[0]['lr']))
    scheduler.step(loss)