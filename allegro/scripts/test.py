import torch
import time
import numpy as np

device = "cuda:0"
weight = torch.randn(32, 1).to(device)
time1 = 0
time2 = 0
time3 = 0
time4 = 0
time5 = 0

for i in range(10000):
    # num = int(np.random.randint(low=0, high=100001, size=1))
    num = 15
    x = torch.randn(num, 32).to(device)
    x.requires_grad_(True)
    idx = torch.arange(num)
    idx = torch.randperm(idx.shape[0]).to(device)
    mask = torch.rand(num).to(device) > 0.5
    torch.cuda.synchronize()

    start = time.time()
    test1 = torch.sum(torch.matmul(x[idx], weight))
    grad1 = torch.autograd.grad(test1, x, create_graph=False)
    torch.cuda.synchronize()
    end = time.time()
    time1 += (end - start)

    start = time.time()
    test2 = torch.sum(torch.matmul(torch.nn.functional.embedding(idx, x), weight))
    grad2 = torch.autograd.grad(test2, x, create_graph=False)
    torch.cuda.synchronize()
    end = time.time()
    time2 += (end - start)

    start = time.time()
    test3 = torch.sum(torch.matmul(torch.index_select(x, 0, idx), weight))
    grad3 = torch.autograd.grad(test3, x, create_graph=False)
    torch.cuda.synchronize()
    end = time.time()
    time3 += (end - start)

    start = time.time()
    test4 = torch.sum(torch.matmul(x[mask], weight))
    grad4 = torch.autograd.grad(test4, x, create_graph=False)
    torch.cuda.synchronize()
    end = time.time()
    time4 += (end - start)

    start = time.time()
    test5 = torch.sum(torch.matmul(torch.masked_select(x, mask.view(-1, 1)).view(-1, x.shape[1]), weight))
    grad5 = torch.autograd.grad(test5, x, create_graph=False)
    torch.cuda.synchronize()
    end = time.time()
    time5 += (end - start)
print("Time of x[idx]:")
print("{} s".format(time1))
print("Time of torch.nn.functional.embedding:")
print("{} s".format(time2))
print("Time of torch.index_select:")
print("{} s".format(time3))
print("Time of x[mask]:")
print("{} s".format(time4))
print("Time of torch.masked_select:")
print("{} s".format(time5))