"""
batch normalization
验证模型时，net.eval()关闭bn，也就是不计算本次数据的动态均值方差，使用之前的移动平均估计的均值方差和参数标准化
验证模型后，net.train()开启bn
"""

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import matplotlib.pyplot as plt


N_SAMPLES = 2000
EPOCH = 10
BATCH_SIZE = 64
LR = 0.001
N_HIDDEN = 8
B_INIT = -2
ACTIVISION = F.relu

x = np.linspace(-5, 10, N_SAMPLES)[:,np.newaxis]
y = np.square(x) + np.random.normal(0, 2, x.shape)
# plt.scatter(x, y)
# plt.show()

test_x = np.linspace(-5, 10, 200)[:,np.newaxis]
test_y = np.square(test_x) + np.random.normal(0, 2, test_x.shape)

train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
test_x, test_y = torch.from_numpy(test_x), torch.from_numpy(test_y)
train_data = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


class BNModel(nn.Module):
    def __init__(self, batch_normalization=False):
        super(BNModel, self).__init__()
        self.do_bn = batch_normalization
        for i in range(N_HIDDEN):
            input_size = 1 if i == 0 else 10
            self.add_module('fc% i' % i, nn.Linear(input_size, 10))
            if self.do_bn:
                self.add_module('bn%i'%i, nn.BatchNorm1d(10))
        self.predict = nn.Linear(10, 1)

    def forward(self, x):
        for i in range(N_HIDDEN):
            x = getattr(self, 'fc%i'%i)(x)
            if self.do_bn:
                x = getattr(self, 'bn%i'%i)(x)
            x = ACTIVISION(x)
        out = self.predict(x)
        return out


net = BNModel()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()

for i in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        prediction = net(batch_x)
        loss = loss_func(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % i == 0:
            net.eval()
            test_loss = loss_func(net(test_x), test_y)
            print(test_loss.item())
            net.train()

