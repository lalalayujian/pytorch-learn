"""
torch的基本数据格式是tensor张量，类似于numpy的ndarrays，支持索引、切片、连接、随机抽样等
也支持多种数学运算，比如加减乘除、张量叉乘点乘、矩阵分解等
从接口的角度来讲，对tensor的操作可分为两类：torch.function, tensor.function
从存储的角度来讲，对tensor的操作又可分为两类：tensor.add, tensor.add_
"""


import torch
import numpy as np


# 根据可选择的大小或数据新建一个tensor
a = torch.FloatTensor([1,2,3])
print('根据list创建：', a)
b = torch.FloatTensor(2,3)  # 或者torch.Tensor(a.size())，指定大小创建时，不会马上分配空间，只会计算内存是否够用，使用到时才分配
print('指定大小创建:', b)
print('size是一个元组：', b.size())
# torch.tensor()和np.array()很类似，传入数据

# 基本函数操作
print('基本函数操作'.center(50, '='))
a = torch.randn(2, 5)
a1 = a.abs()
print('\n不带下划线后缀的abs函数操作，返回新tensor：\n', a)
a2 = a.abs_()
print('带下划线后缀的abs_函数操作，改变原始tensor：\n', a)


print('\n numpy和tensor的转换')
array = np.array([[1, 2], [3, 4]])
tensor = torch.Tensor(array)  # 默认类型一致的话，会共享内存。torch.tensor(array)不管类型是否一致，都进行数据拷贝
tensor2array = tensor.numpy()  # cuda不能直接转换成numpy
print('numpy array:\n', array)
print('torch tensor:\n', tensor)
print('tensor2array:\n', tensor2array)

print('tensor数学运算'.center(50, '='))
tensor = torch.Tensor([[1, 2], [3, 4]])

tensoradd = torch.Tensor.add(tensor, 5)
print('tensoradd:\n', tensoradd)

tensormm = torch.Tensor.matmul(tensor, tensor)
print('tensormm:\n', tensormm)

tensordot = torch.Tensor.dot(tensor.flatten(), tensor.flatten())
print('tensordot:\n', tensordot)

# cpu和gpu之间的转换
gpu_tensor = torch.randn(2, 5).cuda(device=None)  # 返回cpu内存此tensor的一个副本
cpu_tensor = gpu_tensor.cpu()  # 如果在CPU上没有该tensor，则会返回一个CPU的副本
