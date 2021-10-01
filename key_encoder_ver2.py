import torch
from torch import nn
from torch.nn.modules.conv import Conv2d

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.is_channelchanged = False
        if (in_channels != out_channels):
            self.is_channelchanged = True

    def forward(self, x):
        if (self.is_channelchanged):
            channel_scaling = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
            residual = channel_scaling(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class STEM(nn.Module):      # 256x256x3 -> 128x128x64
    def __init__(self):
        super().__init__()

        self.res1 = ResBlock(3, 16, padding=1)
        self.res2 = ResBlock(16, 32, padding=1)
        self.res3 = ResBlock(32, 64, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)

        out = self.avgpool(out)

        return out


class Encoder(nn.Module):   # 128x128x64 -> 64x64x32
    def __init__(self):
        super().__init__()

        self.stem = STEM()

        self.conv1 = Conv3x3(64, 128, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = Conv3x3(128, 256, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = Conv3x3(256, 256, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = Conv3x3(256, 256, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
    
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.stem(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.avgpool(out)

        return out