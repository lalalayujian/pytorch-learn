"""
Pytorch动态图，每次前向传播都重新生成计算图，可不一样。计算图中，只有叶子节点的变量才会保留梯度
如果tensor的属性requires_grad设置为True，则会跟踪针对该tensor的所有操作（操作抽象为Function），构建计算图
完成计算后，调用.backward()来自动计算梯度，保存在.grad属性中，完成反向传播后中间变量的梯度随即释放
停止tensor历史记录的跟踪：.detach()将其与计算历史分离
停止跟踪历史记录（和使用内存）：求导需要缓存许多中间结构，增加额外的内存/显存开销，测试时可以关闭自动求导with torch.no_grad()
"""
import torch

# ============================autograd机制==========================
# 正向传播
weight1 = torch.ones(2, requires_grad=True)  # 创建网络时，参数都默认requires_grad=True
z = 4 * torch.pow(weight1, 2)
weight2 = torch.ones(2, requires_grad=True)
y = z + weight2
loss = y.sum()  # 参数变量是weight
print('loss:', loss)
# 正向传播时，每层数据处理时会生成一个function实例，每层的输出的创造者grad_fn被设置为这个function实例，记录下数据处理流程
print('weight1 creator:', weight1.grad_fn)
print('weight2 creator:', weight2.grad_fn)
print('y creator:', y.grad_fn)

# 反向传播，autograd机制
loss.backward()  # 标量才能backward，因为loss是一个标量
# 反向传播时，就能根据grad_fn跟踪从任何变量到叶节点的路径，自动反向计算梯度并存储在grad属性中
# 只有grad_fn为None的变量(计算流图的叶节点，上面的weight)才能保存grad
print('weight1 grad:', weight1.grad)  # x相当于网络中的 w
print('weight2 grad:', weight2.grad)
print('y grad:', y.grad)

# a.backward()的grad_variables参数的含义: 等于目标函数对a的梯度
# loss.backward()对loss的梯度为1，省略了。y.backward(torch.tensor([1,1]))等价

# ============================梯度清零==========================
# 梯度计算会累加之前的梯度，所以反向传播之前需把梯度清零
x = torch.ones(2, 2, requires_grad=True)
y = x.sum()
y.backward()
print(x.grad)
y.backward()
print(x.grad)
y.backward()
print(x.grad)

x.grad.zero_()
y.backward()
print(x.grad)

# 如果想要使用tensor的值，但又不希望被autograd记录，那么可以使用tensor.data 或者tensor.detach()

# ==================================cuda=========================

x = torch.cuda.FloatTensor(1)
print(x.get_device())
y = torch.FloatTensor(1).cuda()
print(y.get_device())

# 使用训练设备
use_gpu = False  # torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
# net.to(device)
