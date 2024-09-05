import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(weights='DEFAULT')

        # Modify the last fully connected layer to fit the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Add global average pooling layer to adapt to different input sizes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Pass the input through ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Pass through the modified fully connected layer
        x = self.resnet.fc(x)
        return x
