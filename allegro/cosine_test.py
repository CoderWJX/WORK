import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def cosine_sch(x):
    eta_min = 0
    base_lr = 0.08
    T_max = 50
    # last_epoch = 300
    # y = base_lr - (base_lr - eta_min) * (1 + math.cos((x) * math.pi / T_max)) / 2
    y = (1 + math.cos(((x%T_max)) * math.pi / T_max)) / 2
    # y =  base_lr * torch.sigmoid(torch.tensor((x-T_max/2)/10))

    # y = base_lr * (1-math.cos(x.astype(int)/30))

    return y

xpoints = np.arange(0, 300, 1)
out_array = []
for i in range(len(xpoints)):
    out_array.append(cosine_sch(xpoints[i]))
    i += 1

plt.plot(xpoints, out_array, 'b-')
plt.ylabel('corr_coef')
plt.xlabel('training epochs')
plt.show()
