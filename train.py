import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader
from model import Encoder

writer = SummaryWriter()

# show writer loss 
loss_fn = nn.MSELoss()
lr = 1e-2
momentum = 0.9
optimizer = optim.Adagrad()

model = Encoder()

# main Train 
for i, (video_num_t0, video_num_t1, img_t0 , img_t1, annot_t0, annot_t1) in enumerate(train_loader):
    optimizer.zero_grad()
    
    if video_num_t0 != video_num_t1:
        continue

    # when t, Encoder Key 
    t0_key = model(img_t0, annot_t0)
    t1_key = model(img_t1, annot_t1)
    
    loss = loss_fn(t0_key, t1_key)
    loss.backward()
    optimizer.step()

    # show train loss in Tensorboard
    writer.add_scalar("Train Loss", loss, i)