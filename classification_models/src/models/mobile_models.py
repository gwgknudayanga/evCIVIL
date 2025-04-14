import torch.nn as nn
import torchvision.ops as ops
import torchvision.models as models
from torch import Tensor


class Efficientnet(nn.Module):
    def __init__(self, img_channels: int, num_classes: int) -> None:
        super().__init__()
        model = models.efficientnet_b0(pretrained = True)
        new_input_layer = ops.Conv2dNormActivation(in_channels = img_channels,out_channels = 32,kernel_size = 3, stride = 2,padding = 1,bias = False,inplace = True)
        self.efficientdet_modify = nn.Sequential(new_input_layer,*list(model.features.children())[1:],model.avgpool,nn.Dropout(p=0.2, inplace=True))
        self.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes, bias=True))

    def forward(self, x: Tensor) -> Tensor:

        x = self.efficientdet_modify(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)