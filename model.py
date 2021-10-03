import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import crop, scale

from key_encoder_ver2 import Encoder

class Model(nn.Module):
    def __init__(self, first_frame, annot, stem="resnet"):  # fisrt_frame: first frame image tensor / annot: first frame에 대한 annotation (256x256 기준)
        super().__init__()

        self.encoder = Encoder(stem)
        self.memory = self.get_firstkeys(first_frame, self.downsample_annot(annot))
        self.threshold = 1

    def downsample_annot(self, annot):      # 256x256 resolution에서의 annotation을 64x64에 맞게 downsampling한다.
        annot_downsampled = []
        for i in range(len(annot)):
            annot_downsampled.append(((annot[i][0] // 4), (annot[i][1] // 4)))

        return annot_downsampled

    def get_firstkeys(self, first_frame, annot):        # first frame image와 annotation을 주면, 관절에 해당하는 key vector를 뽑아준다.
        first_keymap = self.encoder(first_frame)
        firstkeys = []
        for i in range(len(annot)):
            firstkeys.append(crop(first_keymap, annot[i][0], annot[i][1], 1, 1).cuda())
        
        return firstkeys

    # target key vector와 query key map이 주어지면,
    # query key map의 좌표별 key vector와 target key vector 사이의 eudlidean distance를 계산하여
    # distance matrix를 만들어 리턴한다.
    def calc_distance(self, target, query):
        distance = torch.norm((query - target), dim=1)
        return distance

    def forward(self, query):
        keymap_query = self.encoder(query)      # query image로부터 query key map을 생성한다.
        joints = []

        for i in range(len(self.memory)):       # 각각의 joint에 대해
            distance = self.calc_distance(self.memory[i], keymap_query)     # query key map과의 distance를 계산한 뒤, 
            if (int(torch.min(distance)) > self.threshold):                 # 만약 가장 유사한 key와의 distance가 미리 정의한 threshold를 넘으면
                joints.append(None)
                continue                                                    # 해당 관절이 query에 존재하지 않는다고 판단한다.

            min_idx = int(torch.argmin(distance))       # distance가 최소인 key vector의 index를 구하고,
            min_vertical = min_idx // 64                # index를 기반으로 vertical coordinate와
            min_horizontal = min_idx % 64               # horizontal coordinate를 구한다.

            self.memory[i] = crop(keymap_query, min_vertical, min_horizontal, 1, 1)     # 해당 joint에 대한 memory를 query key 값으로 대체한다.

            # distance map을 x4 upsampling하여 original resolution을 회복시킨 뒤, 가장 distance가 작은 좌표를 joint로 지정한다.
            distance = distance.unsqueeze(0)
            distance_x4 = F.interpolate(distance, scale_factor=4, mode='bilinear')
            scaled_min_idx = int(torch.argmin(distance_x4))
            scaled_min_vertical = scaled_min_idx // 256
            scaled_min_horizontal = scaled_min_idx % 256
            joints.append((scaled_min_vertical, scaled_min_horizontal))
        
        return joints