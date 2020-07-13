"""
数据加载：
    Dataset: 数据集对象，__getitem__方法返回一个样本数据，使得类可以像list一样按下标访问数据，obj[index]等价于obj.__getitem__(index)
        建议：高负载的操作放在__getitem__中，比如加载图片。因为多进程会并行调用__getitem__，将负载高的放在__getitem__中能实现并行加速
    TensorDataset: 包装data_tensor和 target_tensor，成一个DataSet
    DataLoader: 一个可迭代对象，每次将dataset返回的每一条数据拼接成一个batch数据，用以训练模型。可进行shuffle和并行加速
    DataLoader的__iter__方法返回的_DataLoader才是迭代器
    采样器：默认SequentialSample按顺序采样, RandomSample随机采样, WeightSample按样本权重采样（样本不均衡时可用于重采样）
数据处理： torchvision.transfroms包含很多对图片的处理
"""


import torch
import torch.utils.data as Data
import torchvision
from PIL import Image


# ==============================使用Dataset==================================
torch.manual_seed(1)

# hyper parameters
BATCH_SIZE = 32

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
# print(next(iter(loader)))

from collections import Iterator, Iterable

print(isinstance(loader, Iterable))
print(isinstance(loader, Iterator))

# ========mnist
train_data = torchvision.datasets.MNIST(
    root='../Data/mnist/',   # 保存或者提取位置
    train=True,
    transform=torchvision.transforms.ToTensor(),  # DataLoader把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型
                                                  # shape是C*H*W，数值范围缩小为[0.0, 1.0],输入数据为uint8类型时除以255
    download=False
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# print(next(iter(train_loader))[0][0])

# ===============================自己写Dataset================================
# 需要重写三个方法

class MY_MNIST(Data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data, self.target = torch.load(root)

    def __getitem__(self, index):
        """迭代一次获取的数据"""
        img, target = self.data[index], int(self.target[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

train = MY_MNIST('../Data/mnist/MNIST/processed/training.pt', transform=None)
# print(isinstance(train, Iterable))
# print(train[0])
# print(next(iter(train))[0])  # 迭代一次获取img object

if __name__ == '__main__':
    train = MY_MNIST('../Data/mnist/MNIST/processed/training.pt', transform=torchvision.transforms.ToTensor())
    print(train.data.size())
    # print(len(next(iter(Data.BatchSampler(Data.RandomSampler(train), 32, drop_last=False)))))
    train_loader = Data.DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    print(next(iter(train_loader))[0].size())  # data, target
    print(next(iter(train_loader))[1].size())

