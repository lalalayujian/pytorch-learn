"""
可视化预训练模型
    可视化卷积核
    可视化卷积层提取的特征图片
    可视化输入网络的图片的梯度
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
import json
import time


def load_class_name(p_clsnames, p_clsnames_cn):
    """
    加载ImageNet1000对应的类别名称文件
    """
    with open(p_clsnames, 'r') as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, 'r', encoding='utf-8') as f:
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def preprocess_img(p_img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor([0.485, 0.456, 0.406]),
                             torch.tensor([0.229, 0.224, 0.225]))
    ])

    img = Image.open(p_img)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def imshow(tensor_img, title, top5_cn):
    """
    展示图片及预测概率top5类别
    """
    img = tensor_img.squeeze().detach().numpy().transpose((1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img * std + mean

    plt.imshow(img)
    plt.title(title)
    for i in range(5):
        plt.text(5, 10+i*20, top5_cn[i], bbox=dict(fc='yellow'))
    plt.show()


def predict_img(model, path_img):
    """
    预测单张图片及打印图片
    """
    data_dir = '../Data/pretrain_visual_data'
    path_classnames = os.path.join(data_dir, "imagenet1000.json")
    path_classnames_cn = os.path.join(data_dir, "imagenet_classnames.txt")

    # load class name
    class_names, class_names_cn = load_class_name(path_classnames, path_classnames_cn)

    # load img
    img_tensor = preprocess_img(path_img)

    # predict
    output = model(img_tensor)
    p, top1 = torch.max(output, dim=1)
    p, top5 = torch.topk(output, 5, dim=1)

    # visualization
    pred = class_names_cn[top1.item()]
    print('predict label: ', pred)
    top5_cn = [class_names[i] for i in top5.squeeze().numpy()]
    imshow(img_tensor, img_file, top5_cn)


def visualization(model, path_img):
    """
    可视化第一个卷积层的卷积核（核大小较大且属于低级特征，肉眼更容易懂），以及一张猫图片提取出的特征
    """
    # -------------------------------- kernel visualization ---------------------------------
    kernels = model.features[0].weight
    print(kernels.size())

    # 将多张图片拼接成一张
    rgb_krenels = torchvision.utils.make_grid(kernels, nrow=8, normalize=True, scale_each=True)
    rgb_krenels = rgb_krenels.detach().numpy().transpose((1, 2, 0))  # (3, *, * ) -> (*, *, 3)
    plt.imshow(rgb_krenels)
    plt.show()

    # -------------------------------- feature map visualization ------------------------------
    img_tensor = preprocess_img(path_img)

    # conv feature
    convlayer1 = model.features[0]
    fmap_1 = convlayer1(img_tensor)
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    # 每张图片以自己的均值标准差进行标准化
    fmap_1_grid = torchvision.utils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)
    # print(fmap_1_grid)
    plt.imshow(fmap_1_grid.detach().numpy().transpose((1, 2, 0)))
    plt.show()


class ConvFM():
    """
    可视化输入图片的卷积层的输出
        底层的特征越简单，肉眼更能理解，比如边缘
        越深层的特征越复杂抽象，比如轮廓
    """
    def __init__(self, model, path_img, layer_num):
        self.img_tensor = preprocess_img(path_img)
        self.model = model
        self.model.eval()

        # 方法一：利用hook，取出卷积层的输出
        # 注意：feature map的所有数据都为正，已经被激活，不是单纯的卷积，因为nn.ReLU(inplace=True)
        # module = self.model.features[layer_num]
        # if isinstance(module, nn.Conv2d):
        #     self.hook = module.register_forward_hook(self.hook_fn)
        # self.model(self.img_tensor)
        # self.remove()

        # 方法二：直接前向计算到目标层
        x = self.img_tensor
        for i in range(layer_num + 1):
            x = self.model.features[i](x)
            print(x.size())
        self.features = x

    def hook_fn(self, module, input, output):
        self.features = output  # (b, c, w, h)

    def remove(self):
        self.hook.remove()

    def visual(self):

        # self.fmap_grid = torchvision.utils.make_grid(self.features.transpose(0, 1), nrow=8, normalize=True, scale_each=True)
        # print(self.fmap_grid.size())
        # plt.imshow(self.fmap_grid.detach().numpy().transpose((1, 2, 0)), cmap='gray')
        # plt.show()

        fig = plt.figure(figsize=(12, 12))
        self.features = self.features.transpose(0, 1)
        for i in range(self.features.size(0)):
            ax = fig.add_subplot(8, 8, i+1)
            ax.imshow(self.features[i].detach().numpy().squeeze(), cmap='gray')  # 单通道就输入矩阵即可，三维报错
        plt.show()


class GuidedBackprop():
    """
    利用 Guided Backpropagation 可视化算法，通过反向传播，计算feature map对网络输入的梯度，然后归一化梯度，作为图片显示出来
    梯度大的部分，反映了输入图片该区域对目标输出的影响力较大，反之影响力较小。
    即我们可以了解到神经网络做出的判断到底受图片中哪些区域影响，或者说，目标feature map提取的输入图片中哪些区域的特征
    Guided Backpropagation对反向传播中ReLU部分做了微小的调整：
        1.正常反向传播：输入大于0的部分，ReLU层梯度为1。小于等于0时，梯度为0
        2.deconvet: 输出的梯度大于0的部分，ReLU层梯度为1
        3.Guided Backpropagation：输入大于0且输出的梯度大于0的部分，ReLU层梯度为1
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image_reconsturction = None
        self.activation_maps = []
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            """第一个卷积层反向传播时，计算输入图片的梯度并保存（注意：输入图片的requires_grad=True）"""
            self.image_reconsturction = grad_in[0]
            # print(grad_in[0].size(), grad_in[1].size(), grad_in[2].size())  # dx, dw, db

        def forward_hook_fn(module, input, output):
            """保存ReLU层的前向传播输出，反向传播时需要使用"""
            self.activation_maps.append(input[0])
            # self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            """Guided Backpropagation 的ReLU反向传播"""
            # 输入大于0的部分，梯度为1
            grad = self.activation_maps.pop()
            grad[grad > 0] = 1

            # 只保留梯度大于0的部分
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)

            # 创建新的输入端梯度
            new_grad_in = positive_grad_out #* grad
            # ReLU 不含 parameter，输入端梯度是一个只有一个元素的 tuple
            return (new_grad_in, )

        for name, module in self.model.features.named_children():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)

        first_layer = self.model.features[0]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, p_image, target_class):
        input_image = preprocess_img(p_image)
        # 获取输出，注册的forward hook开始起作用
        model_output = self.model(input_image.requires_grad_())
        pred_class = model_output.argmax().item()

        self.model.zero_grad()

        # 生成one-hot向量，作为反向传播的起点
        grad_target_map = torch.zeros(model_output.size(), dtype=torch.float)
        if target_class:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1

        # 反向传播，注册的 backward hook 开始起作用
        model_output.backward(grad_target_map)
        # 得到 target class 对输入图片的梯度，转换成图片格式
        results = self.image_reconsturction.data[0].permute(1, 2, 0).numpy()
        norm_grad = self.normalize(results)
        plt.imshow(norm_grad)
        plt.show()

    def normalize(self, grad_img):
        # # 归一化梯度map，先归一化到 mean=0 std=1
        norm = (grad_img - grad_img.mean()) / grad_img.std()
        # 设置新的均值方差，让梯度map中的数值尽可能接近0，且保证大部分的梯度值为正
        norm = norm * 0.1 + 0.5
        # 把 0，1 以外的梯度值分别设置为 0 和 1
        norm = np.clip(norm, a_min=0, a_max=1)
        return norm


if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    # img_file = 'Golden Retriever from baidu.jpg'
    img_file = 'tiger cat.jpg'
    path_img = os.path.join('../Data/pretrain_visual_data', img_file)

    # predict_img(model, path_img)
    # visualization(model, path_img)

    # feature_map = ConvFM(model, path_img, 0)  # 0,3,6,8,10
    # feature_map.visual()

    # 展示输入图片的梯度
    guided_bp = GuidedBackprop(model)
    guided_bp.visualize(path_img, 450)

models.resnet18()