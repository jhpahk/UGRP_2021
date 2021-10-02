import torch
import torch.nn.functional as F
from torch import nn

import mod_resnet

# input size: 256x256x3

class Conv3x3(nn.Module):   # Bottleneck 3x3
    def __init__(self, in_channels, out_channels, mid_channels=16, stride=1, padding=0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


# ---------- From STCN ---------- #
class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x
# ---------- From STCN ---------- #


class STEM(nn.Module):      # Sten network based on ResNet / 256x256x3 -> 64x64x256
    def __init__(self):
        super().__init__()

        resnet18 = mod_resnet.resnet18()
        
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3

        self.up1 = UpsampleBlock(128, 256, 256)
        self.up2 = UpsampleBlock(64, 256, 256)

    def forward(self, x):       # x: 256x256x3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # x: 64x64x64

        f1 = self.layer1(x)     # f1: 64x64x64
        f2 = self.layer2(f1)    # f2: 32x32x128
        f3 = self.layer3(f2)    # f3: 16x16x256

        out = self.up1(f2, f3)      # out = f2 + f3 (upsampling and skip connection)
        out = self.up2(f1, out)     # out = f1 + out (upsampling and skip connection)

        return out
        

class Encoder(nn.Module):   # 64x64x256 -> 64x64x32
    def __init__(self):
        super().__init__()

        self.stem = STEM()

        self.conv1 = Conv3x3(256, 256, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = Conv3x3(256, 256, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.stem(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        return out