import torch
import bsq_ext
import math

from torch import autograd

class QuanBaseFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = bsq_ext.quant_base_forward(x)
        ctx.save_for_backward(y,x)
        return y
    
    @staticmethod
    def backward(ctx, grad_from_top):
        y,x = ctx.saved_tensors
        grad_x = bsq_ext.quant_base_backward(x,y)
        return grad_from_top * grad_x


if __name__ == '__main__':
    def ctos(a):
        a = torch.clamp(a+0.5,1/256,254/256)
        return torch.log(a/(1-a))
    # a = t.FloatTensor([ctos(0.13),ctos(-0.22),ctos(0.45),ctos(-0.41)]).reshape(2,2,1,1)
    a = torch.rand((2,2,1,1))
    a.requires_grad_(True)
    a1 = a.detach().clone()
    a1.requires_grad_(True)
    y = QuanBaseFunc.apply(a)
    y1 = 2*torch.sigmoid(a1) - 1
    # print(y,y1)
    loss = y.sum()
    loss1 = y1.sum()
    loss.backward()
    loss1.backward()
    print(a.grad,a1.grad)
    assert all(((a.grad-a1.grad).abs()<1e-5).flatten()),'error'

    x = torch.rand((2,2,1,1))
    s0 = bsq_ext.quant_base_init(x)
    s1 = ctos(x)
    print(s1, s0)
    assert all(((s1-s0).abs()<1e-5).flatten()), 'error'
