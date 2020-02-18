import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
testTransform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        normTransform
    ])
trainLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=True, download=True,
                     transform=trainTransform),
        batch_size=128, shuffle=True)
testLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=False, download=True,
                     transform=testTransform),
        batch_size=64, shuffle=False)


if __name__ == "__main__":
    dataiter=iter(trainLoader)
    images,labels=dataiter.next()
    print(images.shape)
    print(labels)
    torchvision.utils.save_image(images[0],"test.jpg")