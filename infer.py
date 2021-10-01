import torch
from key_encoder import Encoder as Encoder_ver1
from key_encoder_ver2 import Encoder as Encoder_ver2

import time

example = torch.rand((1, 3, 256, 256))
annot = [25, 30]

encoder1 = Encoder_ver1()
encoder2 = Encoder_ver2()

start = time.time()
key1 = encoder1(example, annot)
end = time.time()

print(key1.shape)
print(f"time: {end - start}s\n")

start = time.time()
key2 = encoder2(example)
end = time.time()

print(key2.shape)
print(f"time: {end - start}s\n")