"""
ResNet网络，解决了训练极深网络的梯度消失问题
网络中间有很多结构相似的单元，这些重复的单元的共同点就是有个跨层连接的shortcut
将跨层直连的单元称为Residual Block
ResNet网络大体框架一致: conv1+bn1+relu+maxpool + layer1 + layer2 + layer3 + layer4 + AdaptiveAvgPool2d + fc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


resnet = models.resnet34()
print(resnet)


class ResidualBlock(nn.Module):
    """
    实现子Module：Residual Block
    左边是普通卷积网络结构，右边是直连
    若输入输出的通道数不一致，或步长不为1，那直连就需要一个专门的单元将其转成一致，才能相加
    """
    def __init__(self, inchannel, outchannle, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannle, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannle),
            nn.ReLU(),
            nn.Conv2d(outchannle, outchannle, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannle))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    """
    实现主Module：ResNet34
    """
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1))

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer，包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet34()
print(model)
print(model(torch.rand(1, 3, 224, 224)))