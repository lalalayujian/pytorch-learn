import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


class _BaseWrapper():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.handlers = []

    def forward(self, image):
        """前向计算各类的概率"""
        self.logits = self.model(image)  # (1, num_classes)
        self.prob = F.softmax(self.logits, dim=1)
        _, self.pred = torch.max(self.prob, dim=1)
        p, self.top5 = torch.topk(self.prob, 5, dim=1)

        return self.top5  # (1, 5)

    def backward(self, ids):
        """计算idx类向量，并反向传播"""
        self.model.zero_grad()
        one_hot = torch.zeros(self.logits.size(), dtype=torch.float)
        one_hot[0][ids] = 1.0

        self.logits.backward(one_hot, retain_graph=True)  # 保留计算图，方便多次后向传播

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def preprocess_img(self, p_img):
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

    def normalize(self, grad_img):
        """
         归一化梯度map
        :param grad_img:尺寸(h,w,c)
        :return:
        """

        # 先归一化到 mean=0 std=1
        norm = (grad_img - grad_img.mean()) / grad_img.std()
        # 设置新的均值方差，让梯度map中的数值尽可能接近0，且保证大部分的梯度值为正
        norm = norm * 0.1 + 0.5
        # 把 0，1 以外的梯度值分别设置为 0 和 1
        norm = np.clip(norm, a_min=0, a_max=1)
        return norm


class BackPropagation(_BaseWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, image):
        self.image = image.requires_grad_()
        super().forward(self.image)

    def __call__(self, path_image, show_top5=False, index=None):
        """展示预测类别或者top5个类别的反向传播梯度图片"""
        image = self.preprocess_img(path_image)
        self.forward(image)

        if show_top5:
            li_idx = self.top5[0]
        else:
            li_idx = [self.pred] if not index else [index]

        for i in li_idx:  # [281, 450]
            print("show target: {}".format(i))
            self.backward(i)
            grad = self.image.grad.detach().squeeze().numpy().transpose((1, 2, 0))
            grad = self.normalize(grad)  # (224, 224, 3)
            plt.imshow(grad)
            plt.show()


class GuidedBackPropgation(BackPropagation):
    def __init__(self, model):
        super(GuidedBackPropgation, self).__init__(model)
        self.activation_maps = []
        self.each_activation_maps = []

        def forward_hook(module, input, output):
            self.activation_maps.append(output)

        def back_hook(module, grad_in, grad_out):
            # 法一：输入>0且grad_out>0的部分，传回梯度
            if len(self.each_activation_maps) == 0:
                self.each_activation_maps = self.activation_maps.copy()
            grad = self.each_activation_maps.pop().clone()
            grad[grad > 0] = 1

            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad
            # 法二：# grad_in是正常ReLU梯度，in>0的部分有梯度，再进行截断
            # new_grad_in= torch.clamp(grad_in[0], min=0.0)
            return (new_grad_in, )  # ReLU不含参数，输入端梯度是一个只有一个元素的 tuple

        for name, module in self.model.features.named_children():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(back_hook)


class DeConvnet(BackPropagation):
    """对target类别不敏感，不同类别反卷积的结果差不多"""
    def __init__(self, model):
        super(DeConvnet, self).__init__(model)

        def back_hook(module, grad_in, grad_out):
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            return (positive_grad_out, )

        for name, module in self.model.features.named_children():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_backward_hook(back_hook))


class CAM(_BaseWrapper):
    """以全局池化层代替fc层"""
    def __init__(self, model, visual_layers):
        super(CAM, self).__init__(model)
        self.visual_layers = visual_layers
        self.fmap = None

        def forward_hook(module, input, output):
            self.fmap = output
            print("feature map shape: {}".format(output.shape))

        # for name, module in self.model.named_children():
        #     if name == self.visual_layers:
        #         self.handlers.append(module.register_forward_hook(forward_hook))

        self.handlers.append(self.model._modules.get(self.visual_layers).register_forward_hook(forward_hook))

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(CAM, self).forward(image)

    def gen_cam(self):
        """
        依据梯度和特征图，生成gcam图
        """
        fmap = self.fmap.detach().numpy().squeeze()  # [c, h, w]
        weight = self.model.fc.weight.detach().numpy()  # (out, c)
        weight = np.mean(weight, axis=0)  # [c]

        gcam = fmap * weight[:, np.newaxis, np.newaxis]  # [c, h, w]
        gcam = np.sum(gcam, axis=0)  # [h, w]

        # 归一化
        gcam = (gcam - np.min(gcam)) / np.max(gcam)

        # cam = cv2.resize(cam, (224, 224))
        ten_gcam = torch.tensor(gcam[np.newaxis, np.newaxis, :])
        # 双线性插值，上采样
        ten_gcam = F.interpolate(ten_gcam, self.image_shape, mode='bilinear', align_corners=False)
        gcam = ten_gcam.squeeze().numpy()  # (224, 224)

        return gcam

    def gcam_on_image(self, image, mask):
        """
        将gcam和原始图片显示在一起，方便可视化定位目标类别的相关区域
        :param image: ndarray
        :param mask: [h,w]
        :return: cam, dim=[H,W,3], value in 0-1
        """
        # 热力图
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255  # [224, 224, 3]
        heatmap = heatmap[..., ::-1]  # gbr to rgb

        # 恢复raw image
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image *= 255

        gcam = heatmap + np.float32(image) / 255
        gcam = (gcam - np.min(gcam)) / np.max(gcam)

        return gcam

    def __call__(self, path_img, show_top5=False, index=None):
        """展示预测类别或者top5个类别的反向传播梯度图片"""
        image = self.preprocess_img(path_img)
        self.forward(image)

        if show_top5:
            li_idx = self.top5[0]
        else:
            li_idx = [self.pred] if not index else [index]

        for i in li_idx:
            print("show target: {}".format(i))

            gcam_mask = self.gen_cam()
            gcam_img = self.gcam_on_image(image.numpy().squeeze().transpose((1,2, 0)), gcam_mask)

            plt.imshow(gcam_img)
            plt.show()


class GradCAM(_BaseWrapper):
    def __init__(self, model, visual_layers):
        super(GradCAM, self).__init__(model)

        self.visual_layers = visual_layers
        self.fmap = None
        self.grad = None

        def forward_hook(module, input, output):
            self.fmap = output
            print("feature map shape: {}".format(output.shape))

        def backward_hook(module, grad_in, grad_out):
            self.grad = grad_out[0]  # [1, 256, 6, 6]

        for name, module in self.model.features.named_children():
            if name == self.visual_layers and isinstance(module, nn.Conv2d):
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_backward_hook(backward_hook))

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def backward(self, ids):
        """计算类向量并反向传播"""
        self.model.zero_grad()
        class_vec = self.logits[0][ids]  # tensor(15.4265, grad_fn=<SelectBackward>)
        # print(class_vec)
        class_vec.backward(retain_graph=True)

    def gen_cam(self):
        """
        依据梯度和特征图，生成gcam图
        """
        grad = self.grad.detach().numpy().squeeze()  # [c, h, w]
        fmap = self.fmap.detach().numpy().squeeze()  # [c, h, w]

        weight = np.mean(grad, axis=(1, 2))  # [c]
        gcam = fmap * weight[:, np.newaxis, np.newaxis]  # [c, h, w]
        gcam = np.sum(gcam, axis=0)  # [h, w]
        gcam = np.maximum(gcam, 0)  # ReLU

        # 归一化
        gcam = (gcam - gcam.min()) / gcam.max()

        # cam = cv2.resize(cam, (224, 224))
        ten_gcam = torch.tensor(gcam[np.newaxis, np.newaxis, :])
        # 双线性插值，上采样
        ten_gcam = F.interpolate(ten_gcam, self.image_shape, mode='bilinear', align_corners=False)
        gcam = ten_gcam.squeeze().numpy()  # (224, 224)

        return gcam

    def gcam_on_image(self, image, mask):
        """
        将gcam和原始图片显示在一起，方便可视化定位目标类别的相关区域
        :param image: ndarray
        :param mask: [h,w]
        :return: cam, dim=[H,W,3], value in 0-1
        """
        # 热力图
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255  # [224, 224, 3]
        heatmap = heatmap[..., ::-1]  # gbr to rgb

        # 恢复raw image
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image *= 255

        gcam = heatmap + np.float32(image) / 255
        gcam = (gcam - np.min(gcam)) / np.max(gcam)

        return gcam

    def __call__(self, path_img, show_top5=False, index=None):
        """展示预测类别或者top5个类别的Grad CAM图片"""
        image = self.preprocess_img(path_img)

        self.forward(image)

        if show_top5:
            li_idx = self.top5[0]
        else:
            li_idx = [self.pred] if not index else [index]

        for i in li_idx:
            print("show target: {}".format(i))
            self.backward(i)

            gcam_mask = self.gen_cam()
            gcam_img = self.gcam_on_image(image.numpy().squeeze().transpose((1,2, 0)), gcam_mask)

            plt.imshow(gcam_img)
            plt.show()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++主要改进了weight的计算方式，不像GardCAM，一个通道的像素权重一样。此处每个通道的每个像素的权重和梯度大小成正比
    可得到更好的效果，特别是在某一分类的物体在图像上不止一个的情况下
    """
    def __init__(self, model, visual_layers):
        super().__init__(model, visual_layers)

    def gen_cam(self):
        """
        依据梯度和特征图，生成gcam图
        """
        fmap = self.fmap.detach().numpy().squeeze()  # [c, h, w]

        grad = self.grad.detach().numpy().squeeze()  # [c, h, w]
        grad = np.maximum(grad, 0.)  # ReLU

        indicate = np.where(grad > 0, 1., 0.)  # 大于0的位置为1
        norm_factor = np.sum(grad, axis=(1, 2)) # [c]
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0 else 0
        # 梯度大于0的位置：每个通道梯度大于0的总梯度的导数
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [c, h, w]
        print(alpha)
        weight = np.sum(grad * alpha, axis=(1, 2))  # [c]

        gcam = fmap * weight[:, np.newaxis, np.newaxis]  # [c, h, w]
        gcam = np.sum(gcam, axis=0)  # [h, w]

        # 归一化
        gcam = (gcam - np.min(gcam)) / np.max(gcam)

        # cam = cv2.resize(cam, (224, 224))
        ten_gcam = torch.tensor(gcam[np.newaxis, np.newaxis, :])
        # 双线性插值，上采样
        ten_gcam = F.interpolate(ten_gcam, self.image_shape, mode='bilinear', align_corners=False)
        gcam = ten_gcam.squeeze().numpy()  # (224, 224)

        return gcam

    def __call__(self, path_img, show_top5=False, index=None):
        image = self.preprocess_img(path_img)

        self.forward(image)

        if show_top5:
            li_idx = self.top5[0]
        else:
            li_idx = [self.pred] if not index else [index]

        for i in li_idx:
            print("show target: {}".format(i))
            self.backward(i)

            gcam_mask = self.gen_cam()
            gcam_img = self.gcam_on_image(image.numpy().squeeze().transpose((1,2, 0)), gcam_mask)

            plt.imshow(gcam_img)
            plt.show()


class GuidGradCAM(_BaseWrapper):
    """
    GuidGradCAM，既能定位相关区域，也具有细粒度空间梯度可视化
    实现：Guided 和 GradCAM对应元素相乘
    """
    def __init__(self, model, visual_layers):
        super().__init__(model)
        self.gbp = GuidedBackPropgation(model)
        self.gcam = GradCAM(model, visual_layers)

    def __call__(self, path_img, show_top5=False, index=None):
        """展示预测类别或者top5个类别的Guided Grad CAM图片"""
        image = self.preprocess_img(path_img)

        self.gbp.forward(image.clone())
        self.gcam.forward(image.clone())

        if show_top5:
            li_idx = self.top5[0]
        else:
            li_idx = [self.gbp.pred] if not index else [index]

        for i in li_idx:
            print("show target: {}".format(i))
            self.gbp.backward(i)
            grad = self.gbp.image.grad.detach().squeeze().numpy().transpose((1, 2, 0))
            grad = self.normalize(grad)  # (224, 224, 3)

            self.gcam.backward(i)
            gcam_mask = self.gcam.gen_cam()  # (224, 224)

            gcam_gbp = gcam_mask[:, :, np.newaxis] * grad
            gcam_gbp = self.normalize(gcam_gbp)
            plt.imshow(gcam_gbp)
            plt.show()

    def remove_hook(self):
        self.gbp.remove_hook()
        self.gcam.remove_hook()


if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    # img_file = 'Golden Retriever from baidu.jpg'
    img_file = 'tiger cat.jpg'
    path_img = os.path.join('../Data/pretrain_visual_data', img_file)

    # deconv = DeConvnet(model)
    # deconv.visual(image)
    # deconv.remove_hook()

    gbp = GuidedBackPropgation(model)
    gbp(path_img)
    gbp.remove_hook()

    # gcam = GradCAM(model, visual_layers='10')
    # gcam(path_img, show_top5=False)
    # gcam.remove_hook()

    # ggcam = GradCAMPlusPlus(model, visual_layers='10')
    # ggcam(path_img, show_top5=False)
    # ggcam.remove_hook()

    # ggcam = GuidGradCAM(model, visual_layers='10')
    # ggcam(path_img, show_top5=False)
    # ggcam.remove_hook()


    # model = models.resnet18(pretrained=True)
    # cam = CAM(model, visual_layers='layer4')
    # cam(path_img, show_top5=False, index=281)
    # cam.remove_hook()
models.detection.fasterrcnn_resnet50_fpn()
models.vgg16()