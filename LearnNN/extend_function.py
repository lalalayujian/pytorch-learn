"""
pytorch利用Tensor和Function来构建计算图
每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用
Function一般只定义一个操作，因为其无法保存参数，因此适用于激活函数、pooling等操作
拓展autograd，封装自己需要的函数，不仅有前向传播的运算，也需要支持反向传播计算梯度
    1.继承Function类
    2.定义forward方法，参数为前向传播需要的tensor
    3.定义backward方法，参数为标量反向传播到此处的输入梯度
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck
import matplotlib.pyplot as plt

# ==============================封装Function类====================================
print('封装Function类'.center(30, '='))
class MultiplyAdd(Function):            # 继承Function类

    @staticmethod                       # forward和backward都是静态方法，静态方法不能有self
    def forward(ctx, w, x, b):          # ctx上下文对象，可保存前向传播的中间数据
        ctx.save_for_backward(w, x, b)  # ctx保存数据，方便backward，只能保存tensor，其他类型可使用ctx.x=x
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, gradoutput):       # ctx上下文对象，在前向传播保存了数据
        w, x, b = ctx.saved_tensors      # ctx读取参数
        grad_w = gradoutput * x
        grad_x = gradoutput * w
        grad_b = gradoutput
        return grad_w, grad_x, grad_b    # backward输出参数和forward输入参数必须一一对应


# ====================================调用方法==================================
# 类名.apply(参数)
x = torch.ones(1)
w = torch.rand(1, requires_grad = True)
b = torch.rand(1, requires_grad = True)
print('开始前向传播')
z = MultiplyAdd.apply(w, x, b)
print(z)
print('开始反向传播')
# 情况一：
z.backward()                             # z作为标量可直接backward()
print(w.grad, x.grad, b.grad)            # x不需要导数，但中间过程还是会计算，但随后被清空
# 情况二：
x = torch.ones(1)
w = torch.rand(1, requires_grad = True)
b = torch.rand(1, requires_grad = True)
z = MultiplyAdd.apply(w, x, b)          # 会自动保存操作类在grad_fn属性中
print(z.grad_fn)                        # 操作类Function的name+Backward：MultiplyAddBackward
print(z.grad_fn.apply(torch.ones(1)))   # z作为中间变量，调用gard_fn.apply()，参数为dz。在计算中间梯度，buffer并未清空

# ===================================梯度检查=====================================
test_x = torch.tensor(5)
test_w = torch.rand(1, requires_grad = True)
test_b = torch.rand(1, requires_grad = True)
# 采用数值逼近方式检验计算梯度的公式是否正确
print('梯度检查：', gradcheck(MultiplyAdd.apply, (test_w, test_x, test_b), eps=1e-3))

# ==================================高阶导数=======================================
print('高阶导数'.center(30, '='))
x = torch.tensor([5], requires_grad=True, dtype=torch.float32)
y = x ** 2
grad_x = torch.autograd.grad(y, x, create_graph=True)  # create_graph返回变量时是否创建计算图，不创建的话，无法继续求导
print(grad_x)
grad2_x = torch.autograd.grad(grad_x[0], x)
print(grad2_x)

# ==============================其他Function=========================================
class Sigmoid(Function):
    @staticmethod
    def forward(ctx,  input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)  # p
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        output, = ctx.saved_tensors
        gradinput = output * (1 - output) * gradoutput
        return gradinput


x = torch.tensor([1.5], requires_grad=True)
print('梯度检查Sigmoid: ', gradcheck(Sigmoid.apply, (x,), eps=1e-3))

