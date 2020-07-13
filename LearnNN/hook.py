"""
利用pytorch的hook技术可输出模型某部分层的梯度或者输入输出
学习链接：https://zhuanlan.zhihu.com/p/75054200
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================tensor的hook===========================
def tensor_hook(grad):
    print('y的梯度：', grad)

x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
# 注册hook
hook_handle = y.register_hook(tensor_hook)
z = y.sum()
z.backward()

hook_handle.remove()

# =========================Modules的hook=============================


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)
        self.initialize()

    # 特定初始化，方便验证
    def initialize(self):
        self.fc1.weight = nn.Parameter(torch.ones(2, 1))
        self.fc1.bias = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.fc2.weight = nn.Parameter(torch.ones(1, 2))
        self.fc2.bias = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


total_feature_input = []
total_feature_output = []


# 定义前向传播的hook，可获取某个模块的输入和输出，能改变输出变量的值
def hook_forward(module, input, output):
    """在 forward hook 中，input 是 x，而不包括 W 和 b"""
    # print(module)
    # print('input: ', input)
    # print('output: ', output)
    total_feature_input.append(input)
    total_feature_output.append(output)


total_grad_output = []
total_grad_input = []


# 定义反向传播的hook，可获取某个模块的输入和输出梯度，能改变输入变量的梯度
def hook_backward(module, grad_input, grad_output):  # 前向传播角度的输入输出
    """在 backward hook 中，grad_input包括 dW 和 db"""
    # print(module)
    # print('grad_output: ', grad_output)  # 线性模块(dz,)
    # print('grad_input: ', grad_input)  # (db, dx, dw)
    total_grad_output.append(grad_output)
    total_grad_input.append(grad_input)


model = Model()
# 在某些简单模块注册hook，若在包含多个模块的复杂模块绑定，那么只会对模块的最后一个操作起作用（可对model进行测试）
for name, module in model.named_children():  # named_modules还包括model自己
    # 前向传播可获取某个模块的输入和输出，方便使用预训练的模型提取特征
    handle1 = module.register_forward_hook(hook_forward)
    # 后向传播可获取某个模块的梯度输入和梯度输出，反向传播时会自动计算
    handle2 = module.register_backward_hook(hook_backward)

x = torch.tensor([[1.0]])
out = model(x)
handle1.remove()  # 使用完后及时删除
print('saved inputs and outputs'.center(40, '='))
for i in range(len(total_feature_input)):
    print('input: ', total_feature_input[i])
    print('output: ', total_feature_output[i])

loss = out.sum()
loss.backward()
handle2.remove()
print('saved gradoutputs and garadinputs'.center(40, '='))
for i in range(len(total_grad_output)):
    print('grad output: ', total_grad_output[i])
    print('grad input: ', total_grad_input[i])

