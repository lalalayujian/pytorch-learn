import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 64
LR = 0.001
# 使用训练设备
use_gpu = True  # torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

# train_data = torchvision.datasets.MNIST(root='../Data/mnist/',
#                         train=True,
#                         transform=torchvision.transforms.ToTensor(),
#                         download=False)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='../Data/mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255  # 添加通道维度
test_y = test_data.test_labels[:2000]
test_x = test_x.to(device)
test_y = test_y.to(device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(5, 2)  # -> (16, 12, 12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(5, 2)  # -> (32, 4, 4)
        )
        self.out = nn.Linear(32*4*4, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x2.view(x.size(0), -1)
        out = self.out(x3)
        return x1, x2, out

# cnn = CNN().to(device)
# print(cnn)
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()

# for epoch in range(EPOCH):
#     for step, (batch_x, batch_y) in enumerate(train_loader):
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#
#         x1, x2, output = cnn(batch_x)
#         loss = loss_func(output, batch_y)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             test_output = F.softmax(cnn(test_x)[2])
#             test_pred = torch.max(test_output, 1)[1].squeeze()#.numpy()  # 取概率最大的标签，然后压缩成一维向量
#             accuracy = test_pred.eq(test_y).sum().cpu().numpy()/test_pred.size()[0]
#             print('Epoch: ', epoch, '|Step: ',step, '|loss: ', loss.data.cpu().numpy(), '|test accuracy: ', accuracy)
#
# torch.save(cnn, 'mnist_cnn.pk')

cnn = torch.load('mnist_cnn.pk')
x1, x2, test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')
print(x1.size())

x1, x2 = x1[0].cpu().detach().numpy(), x2[0].cpu().detach().numpy()
fig, axes = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(x1[i*4+j, :, :])
plt.show()

fig, axes = plt.subplots(4, 8)
for i in range(4):
    for j in range(8):
        axes[i, j].imshow(x2[i * 4 + j, :, :])

plt.show()
