import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time


EPOCH = 10
BATCH_SIZE = 64
LR = 0.001

device = torch.device('cuda')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
trainset = torchvision.datasets.CIFAR10(
    root='../Data/cifar-10-python/',
    train=True,
    transform=transform,  # 需要看看norm
    download=False)  # 3*32*32
# plt.imshow(train_data[5][0])
# plt.show()
# print(len(trainset))
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # , num_workers=3)
# print(next(iter(train_loader)))

testset = torchvision.datasets.CIFAR10(
    root='../Data/cifar-10-python/',
    train=False,
    transform=transform,
    download=False)
# print(len(testset))
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)  # , num_workers=3)


class LeNet5(nn.Module):
    """LeNet-5卷积网络结构"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(2, stride=2)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(2, stride=2)(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

def train_test(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    train_loss_func = nn.CrossEntropyLoss()  # batch mean loss
    test_loss_func = nn.CrossEntropyLoss(reduction='sum')  # batch sum loss

    start = time.time()
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = net(batch_x.to(device))
            loss = train_loss_func(prediction, batch_y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                batch_loss = []
                eq_sum = 0
                # 因为要统计跟踪每个测试batch的损失，而loss是一个具有autograd的张量，每次保存会GPU内存不足，应分离变量或访问其基础数据
                # 推荐做法：由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
                with torch.no_grad():
                    for test_x, test_y in test_loader:
                        test_pred = net(test_x.to(device))
                        batch_loss.append(test_loss_func(test_pred, test_y.to(device)))
                        test_p = torch.max(F.softmax(test_pred), dim=1)[1]
                        eq_sum += torch.sum(test_p.eq(test_y.to(device))).item()
                test_loss = sum(batch_loss) / 10000
                acc = eq_sum / 10000
                end = time.time()
                print('epoch: %i, | step: %i, |time: %.4f, | train loss: %.4f, | test loss: %.4f, | test accuracy: %.4f'
                      %(epoch, step, end-start, loss.item(), test_loss, acc))
                start = end


if __name__ == '__main__':
    net = LeNet5().to(device)
    print(net)
    train_test(net)
    torch.save(net, 'cifar10.pk')