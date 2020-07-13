"""
Module保存了参数，因此适合于定义一层，如线性层，卷积层，也适用于定义一个网络
拓展nn.Module，封装自己需要的网络层
    1.init方法，定义参数和完成初始化
    2.forward方法，可调用自己定义的对应的op类Function
backward的计算由自动求导机制完成，会调用各个Function的backward得到结果
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck


print('自定义一个线性Function'.center(30, '='))
class LinearFunction(Function):
    """
    自己定义一个线性Function
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = torch.mm(input, weight.t())
        if bias is not None:
            output = torch.add(output, bias)
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input, weight, bias = ctx.saved_tensors
        gradinput = torch.mm(gradoutput, weight)
        gradweight = torch.mm(gradoutput.t(), input)
        gradbias = gradoutput.sum(0)  # bias的梯度是一个向量
        return gradinput, gradweight, gradbias


# 将自定义function封装到函数中
def mylinear(x, weight, bias):
    return LinearFunction.apply(x, weight, bias)


x = torch.tensor([[2]], dtype=torch.float32, requires_grad=True)
weight = torch.ones(2,1, dtype=torch.float32,requires_grad=True)
bias = torch.ones(2, dtype=torch.float32,requires_grad=True)

z = mylinear(x, weight, bias)
print(z.grad_fn)
print(z.grad_fn.apply(torch.ones(1,2)))
print('梯度检查LinearFunc: ', gradcheck(mylinear, (x, weight, bias), eps=1e-3))


print('自定义一个Linear层'.center(30, '='))
class MyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # 定义参数，Parameter类型是tensor的子类，requires_grad默认True
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return mylinear(input, self.weight, self.bias)
        # return LinearFunction.apply(input, self.weight, self.bias)


linearnn = MyLinear(1, 2)
x = torch.tensor([[2]], dtype=torch.float32, requires_grad=True)
z = linearnn(x)
print(z)
print(z.grad_fn)


print('测试线性层的效果'.center(20, '='))
net = nn.Sequential(
    MyLinear(1, 10),
    nn.ReLU(),
    MyLinear(10, 1)
)

optimizer = torch.optim.Adam(net.parameters(), 0.01)
loss_func = nn.MSELoss()

x = torch.rand(300, 1)
y = x ** 2 + torch.randn(x.size())

for i in range(10):
    pred = net(x)
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('loss: ', loss.item())
