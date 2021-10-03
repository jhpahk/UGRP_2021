import torch
import torch.nn.functional as F
from torchvision.transforms.functional import crop

t1 = torch.rand(1, 2, 2, 2)
t2 = torch.rand(1, 2, 2, 2)

# t1 = crop(t1, 수직 좌표, 수평 좌표, 1, 1).reshape(길이)
target = crop(t1, 1, 0, 1, 1)

print(target.shape)
print('\n', t1)
print('\n', t2)
print('\n', target)

distance = t2 - target
print('\n', distance)

distance = torch.norm(distance, dim=1)
print('\n', distance)

min_idx = int(torch.argmin(distance))
min_height = min_idx // 2
min_width = min_idx % 2
print('\n', min_height, min_width)

most_similar = crop(t2, min_height, min_width, 1, 1)
print('\n', most_similar)
print('\n', torch.min(distance))

t5 = torch.rand(1, 2, 2, 2)
print('\n', t5)
t5_x4 = F.interpolate(t5, scale_factor=4, mode='bilinear')
print('\n', t5_x4)

t6 = torch.rand(1, 64, 64)
print('\n', t6.shape)
t6 = t6.unsqueeze(0)
print('\n', t6.shape)