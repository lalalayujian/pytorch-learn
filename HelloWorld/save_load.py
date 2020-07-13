"""
持久化操作：使用的pickle模块
可持久化的对象：dict，tensor, nn.Module, optimizer等
"""


import torch

x = torch.linspace(-1, 1, 200).reshape((-1, 1))
y = x.pow(2) + 0.2 * torch.rand(x.size())

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 10)
        self.predict = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net()

optimizier = torch.optim.SGD(net.parameters(), lr=.2)
loss_func = torch.nn.MSELoss()

for i in range(50):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizier.zero_grad()
    loss.backward()
    optimizier.step()

# 保存整个网络，当网络复杂时，加载可能较慢
torch.save(net, 'net.pkl')
net1 = torch.load('net.pkl')  # 网络保存在GPU的话，需设置map_location='cuda:0'
pred1 = net1(x)

# 只保存模型的参数，速度较快。optimizer也有state_dict，也可保存方便后续继续训练
torch.save(net.state_dict(), 'net_weights.pkl')
net2 = Net()
net2.load_state_dict(torch.load('net_weights.pkl'))
pred2 = net2(x)

# 非严格匹配，忽略掉state_dict中和模型参数不一致的键值对
# net2.load_state_dict('', strict=False)

print(pred1, pred2)
