"""
比较各种optimizer的效果
对于单个optimizer，可为每个子网络的参数设置不同的学习率
"""
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake dataset
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y = x.pow(2) + 0.1 * torch.rand(x.size())

# batch_loader
dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         # num_workers=1
                         )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.predict = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = nn.MSELoss()
losses_his = [[], [], [], []]

for epoch in range(EPOCH):
    print('epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            l_his.append(loss.data.numpy())

plt.plot(range(len(losses_his[0])), losses_his[0], c='b', label='SGD')
plt.plot(range(len(losses_his[1])), losses_his[1], c='g', label='Momentum')
plt.plot(range(len(losses_his[2])), losses_his[2], c='y', label='RMSprop')
plt.plot(range(len(losses_his[3])), losses_his[3], c='r', label='Adam')
plt.legend()
plt.show()


# 不同子网络设置不同的学习率，需包含所有的层，没设置使用最外层的默认学习率
opt_SGD = torch.optim.SGD([
    {'params': net_SGD.hidden.parameters()},
    {'params': net_SGD.predict.parameters(), 'lr': 1e-3}
    ], lr=LR)
print(opt_SGD.param_groups)

# 调整学习率
# 法1：推荐做法--新建优化器，新构建的开销很小，所以可以新建
# 但对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况
old_lr = 0.1
optimizer1 = torch.optim.SGD([
    {'params': net_SGD.hidden.parameters()},
    {'params': net_SGD.predict.parameters(), 'lr': old_lr * 0.1}
    ], lr=LR)

# 法2：手动调整，会保存动量
for param_group in opt_SGD.param_groups:
    param_group['lr'] *= 0.1
print(opt_SGD)

# 法3： 使用学习率调整对象
# 每10step，lr=0.1*lr
scheduler = torch.optim.lr_scheduler.StepLR(opt_SGD, step_size=10, gamma=0.1)
scheduler.step()  # 训练时使用