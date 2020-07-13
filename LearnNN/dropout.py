"""
dropout层可防止过拟合，每次训练随机设置一些神经元为0，不起作用。神经元越多的层，keep_prob越小效果越好
验证模型时，net.eval()关闭dropout
验证模型后，net.train()开启dropout
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


torch.manual_seed(2)

x = torch.linspace(-2, 2, 20).view(-1, 1)
y = x.pow(2) + torch.randn(x.size())
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

test_x = torch.linspace(-2, 2, 20).view(-1, 1)
test_y = test_x.pow(2) + torch.randn(test_x.size())

class FCModel(nn.Module):
    def __init__(self, dropout=False):
        super(FCModel, self).__init__()
        self.dropout = dropout
        self.add_module('fc1', nn.Linear(1, 200))
        self.add_module('fc2', nn.Linear(200, 10))
        self.predict = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.dropout:
            x = nn.Dropout(0.5)(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        if self.dropout:
            x = nn.Dropout(0.8)(x)
        x = nn.ReLU()(x)
        x = self.predict(x)
        return x

nets = [FCModel(), FCModel(dropout=True)]

optimizers = [torch.optim.Adam(nets[0].parameters(), lr=0.01),
              torch.optim.Adam(nets[1].parameters(), lr=0.01),]
loss_func = nn.MSELoss()

loss_train = torch.zeros(2, 50)
loss_test = torch.zeros(2, 50)

for i in range(500):
    for net_i, (net, optimizer) in enumerate(zip(nets, optimizers)):
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss_train[net_i][int(i / 10)] = loss.item()  # 0-dim tensor转换为number
            net.eval()  # 验证模型时，关闭dropout
            test_loss = loss_func(net(test_x), test_y)
            loss_test[net_i][int(i / 10)] = test_loss.item()
            net.train()  # 验证模型后，开启dropout
plt.subplot()
plt.plot(range(50), loss_train.numpy()[0], c='b', label='over train loss')
plt.plot(range(50), loss_train.numpy()[1], c='g', label='drop train loss')
plt.plot(range(50), loss_test.numpy()[0], c='r', label='over test loss')
plt.plot(range(50), loss_test.numpy()[1], c='y', label='drop test loss')
plt.legend()
plt.show()