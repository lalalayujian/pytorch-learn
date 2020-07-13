"""
要使用gpu运算，只需要将数据和实例化神经网络转换为gpu版本
cpu张量转换为gpu张量：x.cuda()
gpu张量转换为cpu张量：x.cpu()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# print(torch.cuda.is_available())

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
x = torch.linspace(-1, 1, 500).reshape((-1, 1))  #.cuda()
y = x.pow(2) + 0.2 * torch.rand(x.size())  #.cuda()
# x = torch.nn.DataParallel(x, device_ids=[0])if torch.cuda.device_count() > 1 else x
# 如果有多个gpu时可以选择上面的语句，例如上面写的时设备0

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """网络的正向传播，边输入数据边搭网络结构"""
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)  #.cuda()
print(net)

# if torch.cuda.device_count() > 1:
#     net = torch.nn.DataParallel(net, device_ids=[0])
#如果多个gpu时，需要修改如上

opti = torch.optim.SGD(net.parameters(), lr=0.5)  # 优化器
loss_func = nn.MSELoss()  # 损失函数

# plt.ion()  # 打开交互模式
# plt.show()

import time
strat = time.time()

for i in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)

    opti.zero_grad()  # 清除上一步更新的参数值
    loss.backward()   # 计算误差反向传递的梯度
    opti.step()       # 利用梯度更新参数

    if i % 5 == 0:
        print(i, loss.data.numpy())
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-')
        plt.text(0.5, 0, 'loss=%.4f'%loss.data.numpy())
        plt.pause(0.1)

end = time.time()
print('time: ', end-strat)
