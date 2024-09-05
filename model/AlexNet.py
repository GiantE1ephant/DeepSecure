import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, dataset_name):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        input_size = (32, 32)
        if dataset_name == 'CIFAR-10':
            input_size = (32, 32)
        if dataset_name == 'MSTAR':
            input_size = (100, 100)

        # 根据输入图像尺寸计算展平后的特征图尺寸
        self._initialize_fc_layers(input_size)

    def _initialize_fc_layers(self, input_size):
        # 计算卷积和池化后的特征图尺寸
        h, w = input_size

        # 通过第一层卷积和池化
        h = (h - 5 + 1) // 2
        w = (w - 5 + 1) // 2

        # 通过第二层卷积和池化
        h = (h - 5 + 1) // 2
        w = (w - 5 + 1) // 2

        # 根据计算得到的尺寸初始化全连接层
        self.fc1 = nn.Linear(16 * h * w, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了batch_size外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

