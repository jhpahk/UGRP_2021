import torch
from torch import optim
import torchvision.transforms as transforms

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torchvision.io import read_image

from PIL import Image

import json

from model import Model


with open("../PoseTrack/annotations_train/000027_bonn_train.json") as f:
    annot_file = json.load(f)


# get all directories of images in the target folder(video)
img_dirs = []
img_ids = []
for i in range(len(annot_file['images'])):
    if annot_file['images'][i]["is_labeled"]:
        img_dirs.append(annot_file['images'][i]['file_name'])
        img_ids.append(annot_file['images'][i]['id'])


# load image and resize to (256, 256)
imgs = {}
for i in range(len(img_dirs)):
    img_resized = Image.open("../PoseTrack/" + img_dirs[i]).resize((256, 256))
    to_tensor = transforms.ToTensor()
    imgs[img_ids[i]] = to_tensor(img_resized).unsqueeze(0)


# load annotations
annots = {}
for i in range(len(annot_file['annotations'])):
    if annot_file['annotations'][i]['image_id'] not in annots:
        annots[annot_file['annotations'][i]['image_id']] = {}
    image_id = annot_file['annotations'][i]['image_id']

    if annot_file['annotations'][i]['track_id'] not in annots[image_id]:
        annots[image_id][annot_file['annotations'][i]['track_id']] = []
    track_id = annot_file['annotations'][i]['track_id']

    for j in range(0, len(annot_file['annotations'][i]['keypoints']), 3):
        if annot_file['annotations'][i]['keypoints'][j + 2] == 0:
            annots[image_id][track_id].append(None)
        else:
            x = annot_file['annotations'][i]['keypoints'][j] // 5           # 1280 -> 256
            y = annot_file['annotations'][i]['keypoints'][j + 1] // 2.8125  # 720 -> 256
            annots[image_id][track_id].append((int(x), int(y)))


model = Model()
# model.cuda()

# compute loss and do backpropagation
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(len(imgs) - 1):
    print(f"{i + 1} / {len(imgs) - 1}")

    prev = img_ids[i]
    query = img_ids[i + 1]

    prev_targets = list(annots[prev].keys())
    query_targets = list(annots[query].keys())

    target_consistent = list(set(prev_targets) & set(query_targets))

    loss = 0
    for target in target_consistent:
        prev_keys = model.get_keys(imgs[prev], annots[prev][target])
        query_keys = model.get_keys(imgs[query], annots[query][target])

        for k in range(len(prev_keys)):
            if prev_keys[k] == None or query_keys[k] == None:
                continue

            loss += torch.norm(prev_keys[k] - query_keys[k])
    
    print(f"loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
