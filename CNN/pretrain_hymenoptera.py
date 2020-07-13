"""
迁移学习
方法一：加载预训练模型并重置最终的全连接层，全部网络进行微调
方法二：加载预训练模型关闭反向传播，将其视为固定的特征提取器，重置最终的全连接层，对全连接层进行微调
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import time


plt.subplots()
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCH = 10
LR = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '../Data/hymenoptera_data'

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]))
])

img_sets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                    transform=transform)
            for x in ['train', 'val']}
data_loaders = {x: DataLoader(img_sets[x], batch_size=BATCH_SIZE, shuffle=True)
                for x in ['train', 'val']}
dataset_sizes = {x: len(img_sets[x]) for x in ['train', 'val']}

# 可视化一批数据
# batch_imgs, label = next(iter(data_loaders['train']))
# new_img = torchvision.utils.make_grid(batch_imgs)
# plt.imshow(new_img.numpy().transpose((2, 1, 0)))
# plt.show()


def train_model(model, optimizer, criterion, scheduler, num_epochs=20):
    """
    训练并保存val acc最优的模型
    """
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'val':
                model.eval()
            else:
                scheduler.step()
                model.train()

            runing_loss = 0.0
            runing_corrects = 0

            for step, (data, label) in enumerate(data_loaders[phase]):
                data = data.to(device)
                label = label.to(device)

                with torch.set_grad_enabled(phase == 'train'):  # train时才能使用梯度
                    output = model(data)
                    loss = criterion(output, label)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if phase == 'train':
                    print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

                runing_loss += loss.item() * data.size(0)
                _, preds = torch.max(output.cpu(), 1)  # 返回值和索引
                runing_corrects += torch.sum(torch.eq(preds, label.cpu())).item()

            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_corrects / dataset_sizes[phase]
            print('{} | loss: {:.4f} | Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model = model.load_state_dict(best_model_wts)
    return model


def finetune_all():
    """
    加载预训练模型并重置最终的全连接层，全部网络进行微调
    """
    model_ft = models.resnet18(pretrained=True)
    num_furs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_furs, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)
    # 学习率调整对象
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model_ft, optimizer, criterion, exp_lr_scheduler, num_epochs=15)


def finetune_fc():
    """
    加载预训练模型关闭反向传播，将其视为固定的特征提取器，重置最终的全连接层，对全连接层进行微调
    """
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_features = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_features, 2)
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_conv.parameters(), lr=LR, momentum=0.9)
    # 学习率调整对象
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model_conv, optimizer, criterion, exp_lr_scheduler, num_epochs=15)

nn.AdaptiveAvgPool2d
finetune_all()  # 效果更好
# finetune_fc()

