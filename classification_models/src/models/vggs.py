import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, List


class VGG(nn.Module):
    def __init__(self, img_channels: int, num_classes: int) -> None:
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))
    

class VGG16(VGG):
    def __init__(self, img_channels: int, num_classes: int) -> None:
        super().__init__(img_channels=img_channels, num_classes=num_classes)
        self.conv_layers = nn.Sequential(
            ConvBlock(img_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG9(VGG):
    def __init__(self, img_channels: int, num_classes: int) -> None:
        super().__init__(img_channels, num_classes)
        self.conv_layers = nn.Sequential(
            ConvBlock(img_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG11(VGG):
    def __init__(self, img_channels: int, num_classes: int) -> None:
        super().__init__(img_channels, num_classes)
        self.conv_layers = nn.Sequential(
            ConvBlock(img_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
