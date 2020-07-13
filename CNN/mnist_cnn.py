import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time


start = time.time()

EPOCH = 1
LR = 0.001
BATCH_SIZE = 64
DOWNLOAD_MNIST  = False  # 如果已经下载好mnist数据就写上 False

# 使用训练设备
use_gpu = False  # torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

train_data = torchvision.datasets.MNIST(
    root='../Data/mnist/',   # 保存或者提取位置
    train=True,
    transform=torchvision.transforms.ToTensor(),  # DataLoader把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型
                                                  # shape是C*H*W，数值范围缩小为[0.0, 1.0],输入数据为uint8类型时除以255
    download=DOWNLOAD_MNIST
)
# plt.imshow(train_data.data.numpy()[0])
# plt.show()

# batch generate
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print(next(iter(train_loader))[0][0])

test_data = torchvision.datasets.MNIST(root='../Data/mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255  # 添加通道维度
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2  # same convolution: p=(kernel_size-1)/2
            ),  # ->(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # ->(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # ->(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # ->(32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平 ->(m, 32*7*7)
        out = self.out(x)
        return out

cnn = CNN().to(device)
print(cnn)

# optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.99))
# 网络内部没有softmax激活，此处就用CrossEntropyLoss。如果LogSoftmax转换成了log-probabilities，就使用NLLLoss
loss_func = nn.CrossEntropyLoss()  # torch的y不需要onehot，内部自动完成onehot

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = cnn(batch_x)
        loss = loss_func(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            test_output = F.softmax(cnn(test_x))
            test_pred = torch.max(test_output, 1)[1].data.squeeze().numpy()  # 取概率最大的标签，然后压缩成一维向量
            accuracy = (test_pred == test_y.data.numpy()).sum()/len(test_pred)
            print('Epoch: ', epoch, '|Step: ',step, '|loss: ', loss.data.cpu().numpy(), '|test accuracy: ', accuracy)

print(time.time()-start)
# ----------------预测-------------------------------
# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')
