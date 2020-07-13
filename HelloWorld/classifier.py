import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation


n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.FloatTensor)

# 快速搭建网络
nn = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),  # 2分类可以使用一个神经元输出，也可以使用2个神经元，相应的损失函数需要改变
    torch.nn.Sigmoid()
)
print(nn)

optimizer = torch.optim.SGD(nn.parameters(), lr=0.02)
loss_func = torch.nn.BCELoss()  # Binary Cross Entropy

plt.ion()
plt.show()

for i in range(50):
    out = nn(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        print(loss.data.numpy())
        prediction = out >= 0.5
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        acc = sum(pred_y==target_y)/len(target_y)
        print(acc)

        plt.cla()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y)
        plt.text(0.5, 0, 'accuracy: %.4f'%acc)
        plt.pause(0.5)

"""
# 保存动态图
ims = []  # 用于保存
fig = plt.figure()

for i in range(50):
    out = nn(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        print(loss.data.numpy())
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        acc = sum(pred_y==target_y)/len(target_y)
        print(acc)
        
        im = plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y).findobj()
        ims.append(im)
ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000)
ani.save('test.gif', writer='pillow')
"""