import torch
from torch import nn

from torchvision.transforms.functional import crop, pad

# Assumption: input channel is 256 (input feature map is 64x64x256)

class Conv3x3(nn.Module):   # Bottleneck 3x3
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super().__init__()

        self.mid_channels = 16

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class Encoder_1x1(nn.Module):   # Assume input size is 1x1x256
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        return out


class Encoder_5x5(nn.Module):   # Assume input size is 5x5x256
    def __init__(self):
        super().__init__()
        self.conv1 = Conv3x3(256, 256)                  # output: 3x3x256
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv2 = nn.Conv2d(256, 4, kernel_size=3)   # output: 1x1x4

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out


class Encoder_9x9(nn.Module):     # Assume input size is 9x9x256
    def __init__(self):
        super().__init__()
        '''
        self.conv1 = Conv3x3(256, 256, stride=2, padding=1)     # output: 9x9x256
        self.bn1 = nn.BatchNorm2d(num_features=256)
        '''

        self.conv1 = Conv3x3(256, 256, stride=2, padding=1)     # output: 5x5x256
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv2 = Conv3x3(256, 256)                          # output: 3x3x256
        self.bn2 = nn.BatchNorm2d(num_features=256)

        self.conv3 = nn.Conv2d(256, 8, kernel_size=3)           # output: 1x1x8

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        return out


class Encoder_17x17(nn.Module):     # Assum input size is 17x17
    def __init__(self):
        super().__init__()
        '''
        self.conv1 = Conv3x3(256, 256, stride=2, padding=1)     # output: 17x17x256
        self.bn1 = nn.BatchNorm2d(num_features=256)
        '''

        self.conv1 = Conv3x3(256, 256, stride=2, padding=1)     # output: 9x9x256
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv2 = Conv3x3(256, 256, stride=2, padding=1)     # output: 5x5x256
        self.bn2 = nn.BatchNorm2d(num_features=256)

        self.conv3 = Conv3x3(256, 256)                          # output: 3x3x256
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.Conv2d(256, 16, kernel_size=3)          # output: 1x1x16

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

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


class STEM(nn.Module):      # 256x256x3 -> 64x64x256
    def __init__(self):
        super().__init__()

        self.res1 = ResBlock(3, 16, padding=1)
        self.res2 = ResBlock(16, 32, padding=1)
        self.res3 = ResBlock(32, 64, padding=1)
        self.res4 = ResBlock(64, 128, padding=1)
        self.res5 = ResBlock(128, 256, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)

        out = self.res4(out)
        out = self.avgpool(out)

        out = self.res5(out)
        out = self.avgpool(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = STEM()

        self.encoder_1x1 = Encoder_1x1()
        self.encoder_5x5 = Encoder_5x5()
        self.encoder_9x9 = Encoder_9x9()
        self.encoder_17x17 = Encoder_17x17()

    def crop_tensors(self, x, annot):
        cropped_tensors = []
        cropped_tensors.append(crop(x, annot[0], annot[1], 1, 1))                     # 1 x 1
        cropped_tensors.append(crop(x, (annot[0] - 2), (annot[1] - 2), 5, 5))         # 5 x 5
        cropped_tensors.append(crop(x, (annot[0] - 4), (annot[1] - 4), 9, 9))         # 9 x 9
        cropped_tensors.append(crop(x, (annot[0] - 8), (annot[1] - 8), 17, 17))       # 17 x 17

        return cropped_tensors

    def forward(self, x, annot):
        out = self.stem(x)

        cropped = self.crop_tensors(out, annot)
        
        key_1x1 = self.encoder_1x1(cropped[0])
        key_5x5 = self.encoder_5x5(cropped[1])
        key_9x9 = self.encoder_9x9(cropped[2])
        key_17x17 = self.encoder_17x17(cropped[3])

        key = torch.cat([key_1x1, key_5x5, key_9x9, key_17x17], dim=1)

        return key