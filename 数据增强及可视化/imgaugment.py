import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
from matplotlib import pyplot as plt
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#展示经过数据增强后的图片
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

#这个函数对输入图像img多次运行图像增广方法aug并展示所有的结果。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

if __name__ == '__main__':

    img=Image.open('./flower.jpg')
    plt.imshow(img)
    # 随机水平翻转
    apply(img, torchvision.transforms.RandomHorizontalFlip())
    # 随机垂直翻转
    apply(img, torchvision.transforms.RandomVerticalFlip())
    #随机裁剪
    #在下面的代码里，我们每次随机裁剪出一块面积为原面积 10%∼100% 的区域，
    #且该区域的宽和高之比随机取自 0.5∼2 ，然后再将该区域的宽和高分别缩放到200像素。
    # 若无特殊说明，本节中 a 和 b 之间的随机数指的是从区间 [a,b] 中随机均匀采样所得到的连续值。
    shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug)


    # 另一类增广方法是变化颜色。
    # 我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
    # 在下面的例子里，我们将图像的亮度随机变化为原图亮度的 50% ( 1−0.5 ) ∼150% ( 1+0.5 )。

    # 改变明亮度
    apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

    # 随机变化图像的色调
    apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

    # 随机变化图像的对比度
    apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0))

    # 随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug)

    # 叠加多个增强方法
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)


    plt.show()