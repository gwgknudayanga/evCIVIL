import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, List


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super().__init__()
        self.expansion = expansion
        self.downsample = downsample

        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        bn1 = nn.BatchNorm2d(out_channels)
        
        conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.block = nn.Sequential(
            conv1,
            bn1,
            self.relu,
            conv2,
            bn2,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.block(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000,
        expantion: int = 1,
        layers: List[int] = [2, 2, 2, 2],
    ) -> None:
        super().__init__()
        self.expansion = expantion

        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))

        self.in_channels = out_channels * self.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # The spatial dimension of the final layer's feature map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class ResNet18(ResNet):
    def __init__(self, img_channels: int, num_classes: int):
        super().__init__(
            img_channels=img_channels,
            block=BasicBlock,
            num_classes=num_classes,
            layers=[2, 2, 2, 2],
        )


class ResNet34(ResNet):
    def __init__(self, img_channels: int, num_classes: int):
        super().__init__(
            img_channels=img_channels,
            block=BasicBlock,
            num_classes=num_classes,
            layers=[3, 4, 6, 3],
        )