import torch
# from key_encoder import Encoder as Encoder_ver1
from key_encoder_ver2 import Encoder as Encoder_ver2

import time

from ptflops import get_model_complexity_info

# annot = [25, 30]

# encoder1 = Encoder_ver1()
encoder2 = Encoder_ver2()
encoder2.cuda()

'''
start = time.time()
key1 = encoder1(example, annot)
end = time.time()

print(key1.shape)
print(f"time: {end - start}s\n")
'''

t = 0

for i in range(100):
    example = torch.rand((1, 3, 256, 256))
    example = example.cuda()

    start = time.time()
    key2 = encoder2(example)
    end = time.time()

    t += (end - start)

print(key2.shape)
print(f"Average FPS: {1 / (t / 100)}\n")

macs, params = get_model_complexity_info(encoder2, (3, 256, 256), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


'''
stem = STEM()
example = torch.rand(1, 3, 256, 256)
stem(example)
'''