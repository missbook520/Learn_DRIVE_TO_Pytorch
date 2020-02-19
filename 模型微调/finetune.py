import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import time
import sys
import PIL
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])


def train_fine_tuning(net, optimizer,train_data,test_data, batch_size=128, num_epochs=5):
    train_iter = DataLoader(train_data,batch_size, shuffle=True)
    test_iter = DataLoader(test_data,batch_size)
    loss = torch.nn.CrossEntropyLoss()
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


if __name__ == '__main__':
    train_imgs = ImageFolder('./hotdog/train',transform=train_augs)
    test_imgs = ImageFolder('./hotdog/test',transform=test_augs)
    #hotdogs = [train_imgs[i][0] for i in range(8)]
    #not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    #show_images(hotdogs + not_hotdogs, 4, 4, scale=1.4)

    # 使用预训练模型时，一定要和预训练时做同样的处理
    # 如果你使用的是torchvision的models
    # 那就要求: All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    pretrained_net = models.resnet18(pretrained=False)
    pretrained_net.load_state_dict(torch.load('./resnet18-5c106cde.pth'))

    # print(pretrained_net)
    print(pretrained_net.fc)

    # 修改预训练模型的部分参数
    pretrained_net.fc = nn.Linear(512, 2)
    print(pretrained_net.fc)

    #将全连接层和卷积层的参数分离，便于设置不同的学习率
    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    #将全连接层的学习率设置为原来的十倍
    lr = 0.01
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr, weight_decay=0.001)

    train_fine_tuning(pretrained_net,optimizer,train_imgs,test_imgs)


    plt.show()