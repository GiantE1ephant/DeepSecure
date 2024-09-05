import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 数据加载函数
def load_data(dataset_name, batch_size=32):
    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 如果你的 MSTAR 数据集也是 RGB 图像，这个变换适用
    ])

    if dataset_name == 'CIFAR-10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    elif dataset_name == 'MSTAR':
        train_set = ImageFolder(root='./data/MSTAR/train', transform=transform)
        test_set = ImageFolder(root='./data/MSTAR/test', transform=transform)
    else:
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 检查数据加载
def check_data_loader(data_loader):
    for images, labels in data_loader:
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        break

