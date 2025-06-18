import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

num_basis = 8
r_max = float(6)
prefactor = 2.0 /r_max

# bessel_weights = (
#         torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
# )
bessel_weights = torch.tensor([3.8925,5.8998,8.5938,12.0237,14.1053,18.4205,20.8414,23.966])
mean = torch.tensor([0.1029, 0.0787, 0.093, 0.0828, 0.0907, 0.0842, 0.0896, 0.0849])
# mean2 = torch.tensor([0.1529, 0.0787,0.093,0.0828,0.0907,0.0842,0.0896,0.0849])
std = torch.tensor([17.7699, 6.6333, 5.3084, 4.3652, 3.8832, 3.4787, 3.2128, 2.9764])
# std2 = torch.tensor([15, 5, 5, 4, 3.8832, 3.4787,3.2128,2.9764])

def _poly_cutoff(x,  p: float = 6.0):
    x = x /6.0

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * np.power(x, p))
    out = out + (p * (p + 2.0) * np.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * np.power(x, p + 2.0))

    return out * (x < 1.0)


def bessel(x):
    """
    Evaluate Bessel Basis for input x.

    Parameters
    ----------
    x : torch.Tensor
        Input
    """
    # if x < 1.5:
    #     x = (x * 0.3) + 1.4
        # std = std * 0.8
    # x = x + 2
    numerator = torch.sin(bessel_weights * x / r_max)
    res = prefactor * (numerator / x)
    res = (res - mean) * std
    # res = (res - mean) * std + 1/8
    return res

def bessel_v2(x):
    """
    Evaluate Bessel Basis for input x.

    Parameters
    ----------
    x : torch.Tensor
        Input
    """
    #v1
    # if x < 1.7:
    #     x = (x * 0.15) + 1.5
    # std = std * 0.8

    x = (x * 0.2) + 2
    # x = x + 2

    # v4
    # x = x + 4

    numerator = torch.sin(bessel_weights * x / r_max)
    res = prefactor * (numerator / x)
    res = (res - mean) * std

    #v2
    # res = (res - mean) * std + 1/8
    #v3
    # res = (res - mean) * std + 1/4

    #v5
    # res[res > 0] = -res[res > 0]

    return res

xpoints = np.arange(0, 6, 0.01)
bess =[]
out_array = []
out_array_v2 =[]
zero = []
poly = []
# bess_dim = []
# bess_dim =torch.tensor([])
for i in range(len(xpoints)):
    res = bessel(xpoints[i]) * _poly_cutoff(xpoints[i])
    res2 = bessel_v2(xpoints[i]) * _poly_cutoff(xpoints[i])
    if i == 0:
        bess_dim = res2.unsqueeze(0)

    else:
        bess_dim = torch.cat((bess_dim, res2.unsqueeze(0)), 0)


    out_array.append(sum(res))
    out_array_v2.append(sum(res2))
    bess.append(sum(bessel(xpoints[i])))
    zero.append(0)
    poly.append(_poly_cutoff(xpoints[i]))
    i += 1

# bess_dim = np.array(bess_dim)
for idx, y_plot in enumerate(bess_dim.T):
    plt.plot(xpoints, y_plot, label=f'radical_basis_{idx}')

plt.plot(xpoints, out_array, 'b-', label=f'embedding sum')
plt.plot(xpoints, out_array_v2, 'm--', label=f'embedding sum v2')

plt.plot(xpoints, zero, 'r--', label=f'zero line')
plt.plot(xpoints, poly, 'g--', label=f'ploy envelope')
# plt.plot(xpoints, bess, 'c--', label=f'original bessl')
plt.axvline(x=1.7, c='grey', ls='--')
plt.axvline(x=1.18, c='grey', ls='--')

plt.ylim((-4.5, 3.5))
plt.ylabel('bessel basis function')
plt.xlabel('r')
plt.legend(loc='best')
plt.show()
