import torch
import torch.nn.functional as F
from torch import nn

import mod_resnet
import ShuffleNetV2

# input size: 256x256x3

class Conv3x3(nn.Module):   # Bottleneck 3x3
    def __init__(self, in_channels, out_channels, mid_channels=8, stride=1, padding=0):
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


class ResBlock(nn.Module):      # ResBlock with bottleneck
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = Conv3x3(indim, outdim, padding=1)
 
        self.conv1 = Conv3x3(indim, outdim, padding=1)
        self.conv2 = Conv3x3(outdim, outdim, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = Conv3x3(skip_c, up_c, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class STEM_ResNet(nn.Module):      # Sten network based on ResNet18 / 256x256x3 -> 64x64x256
    def __init__(self):
        super().__init__()

        resnet18 = mod_resnet.resnet18()
        
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        # self.layer3 = resnet18.layer3

        # self.up1 = UpsampleBlock(128, 256, 256)
        self.up2 = UpsampleBlock(64, 128, 256)

    def forward(self, x):       # x: 256x256x3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # x: 64x64x64

        f1 = self.layer1(x)     # f1: 64x64x64
        f2 = self.layer2(f1)    # f2: 32x32x128
        # f3 = self.layer3(f2)    # f3: 16x16x256

        # out = self.up1(f2, f3)      # out = f2 + f3 (upsampling and skip connection)
        out = self.up2(f1, f2)     # out = f1 + out (upsampling and skip connection)

        return out


class STEM_ShuffleNet(nn.Module):      # Stem Network based on ShuffleNetV2 / 256x256x3 -> 64x64x256
    def __init__(self):
        super().__init__()

        shufflenet = ShuffleNetV2.shufflenet_v2_x1_0(pretrained=True)
        
        self.conv1 = shufflenet.conv1
        self.maxpool = shufflenet.maxpool
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 = shufflenet.stage4

        self.up1 = UpsampleBlock(232, 464, 256)
        self.up2 = UpsampleBlock(116, 256, 256)
        self.up3 = UpsampleBlock(24, 256, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        # print(out.shape)

        f1 = self.stage2(x)
        # print(out.shape)
        f2 = self.stage3(f1)
        # print(out.shape)
        f3 = self.stage4(f2)
        # print(out.shape)

        out = self.up1(f2, f3)
        out = self.up2(f1, out)
        out = self.up3(x, out)

        return out


class Encoder(nn.Module):   # 64x64x256 -> 64x64x32
    def __init__(self, stem="resnet"):
        super().__init__()

        stem = stem.lower()

        if stem == "resnet":
            self.stem = STEM_ResNet()
        elif stem == "shufflenet":
            self.stem = STEM_ShuffleNet()
        else:
            raise ValueError(stem)

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