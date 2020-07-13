"""
利用tensorboard可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# 实例化，指定path
writer = SummaryWriter('runs/regression_graph')

x = torch.linspace(-1, 1, 500).reshape((-1, 1))
y = x.pow(2) + 0.2 * torch.rand(x.size())


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


net = Net(1, 10, 1)
print(net)

opti = torch.optim.SGD(net.parameters(), lr=0.5)  # 优化器
loss_func = nn.MSELoss()  # 损失函数


for epoch in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)

    opti.zero_grad()  # 清除上一步更新的参数值
    loss.backward()   # 计算误差反向传递的梯度
    opti.step()       # 利用梯度更新参数

    # 保存loss和epoch数据
    writer.add_scalar('loss', loss.item(), epoch)

# 将model保存为graph
writer.add_graph(net, x)

writer.close()

# writer.add_image(tag, img)

# ======启动tensorboard=======
# cd 到runs
# tensorboard --logdir=regression_graph
# 打开 http://localhost:6006