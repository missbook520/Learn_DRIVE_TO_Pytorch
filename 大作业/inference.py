import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import math
#from torchvision import transforms
import transforms
from resnet import *
from WideRes import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_data_fashion_mnist(batch_size, root='./dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        # normalize = transforms.Normalize(mean=[mean], std=[std])
        # train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
        #                                  transforms.RandomHorizontalFlip(),
        #                                  #transforms.RandomRotation(10),
        #                                  transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        #                                  transforms.ToTensor(),
        #                                  normalize,
        #                                  transforms.RandomErasing()])
        train_augs = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(probability=0, sh=0.4, r1=0.3, mean=[0.4914]),
        ])
        test_augs = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    #net.train()  # 改回训练模式
    return acc_sum / n




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='./dataset', use_normalize=True)

print('加载最优模型')
net= resnet(num_classes=10,
            depth=32,)
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(torch.load('./model_best.pth')['state_dict'])
print(type(net))
print("best acc:",torch.load('./model_best.pth')['best_acc'])
test_acc = evaluate_accuracy(test_iter, net)
print('test acc %.4f'
              % (test_acc))
print('inference测试集')
#net.eval()
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))