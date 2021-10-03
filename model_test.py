import torch
from model import Model

import time


example_f = torch.rand(1, 3, 256, 256)
annot = [(22, 53), (125, 177), (55, 100), (222, 111), (5, 10), (22, 53), (125, 177), (55, 100), (222, 111), (5, 10), (22, 53), (125, 177), (55, 100), (222, 111), (5, 10)]

model = Model(example_f, annot)
model.cuda()

t = 0
for i in range(100):
    example_q = torch.rand(1, 3, 256, 256).cuda()
    
    start = time.time()
    joints = model(example_q)
    end = time.time()

    t += (end - start)

print(f"Average Time: {t / 100}s")
print(f"Average FPS: {100 / t}")