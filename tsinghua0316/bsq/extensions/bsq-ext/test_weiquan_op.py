import torch
import bsq_ext
import math

from torch import autograd

class QuanFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, intervals, neg, pos):
        y,mask = bsq_ext.weiquan_forward(x,s,intervals,neg,pos)
        # print(mask)
        ctx.save_for_backward(y,x,s,mask)
        ctx.neg = neg
        ctx.pos = pos
        ctx.intervals= intervals
        return y
    
    @staticmethod
    def backward(ctx, grad_from_top):
        y,x,s,mask = ctx.saved_tensors
        grad_x, grad_s = bsq_ext.weiquan_backward(x,s,y,mask,ctx.intervals,ctx.neg,ctx.pos)
        return grad_from_top * grad_x, grad_s, None, None, None

class QuanFunc2(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, neg, pos):
        x_bar = x / s
        mask = (x_bar < neg - 0.5).float() + (x_bar > pos+0.5).float()
        # print(mask)
        x_bar = torch.clamp(x_bar, neg, pos)
        y = x_bar.round() * s
        # print(y)
        ctx.save_for_backward(y, x, s, mask)
        return y
    
    @staticmethod
    def backward(ctx, grad_from_top):
        y,x,s,mask = ctx.saved_tensors
        dr = (1-mask) * (y - x)**2 / s
        dc = mask * y * (y-x)/s
        ds = dr + dc
        # print(ds)
        if s.dim() == 1:
            grad_s = ds.sum().reshape(s.shape)/x.numel()
        else:
            grad_s = ds.sum(dim=list(range(1,grad_from_top.dim())), keepdim=True)*x.shape[0]/x.numel()

        return grad_from_top, grad_s, None, None


if __name__ == '__main__':
    # def ctos(a):
    #     return math.log((0.5+a)/(0.5-a))
    # a = t.FloatTensor([ctos(0.13),ctos(-0.22),ctos(0.45),ctos(-0.41)]).reshape(2,2,1,1)
    a = torch.rand((2,2,1,1))
    a.requires_grad_(True)
    a1 = a.detach().clone()
    a1.requires_grad_(True)
    s = torch.ones((2,1,1,1,),requires_grad=True)
    s1 = torch.ones((2,1,1,1,),requires_grad=True)
    neg = 0
    pos = 6
    y = QuanFunc.apply(a,s,10,neg,pos)
    y1 = torch.sigmoid(a1) - 0.5
    # print(y1)
    
    y1 = QuanFunc2.apply(y1,s1/10,neg,pos)
    y1 = 2*y1
    # print(y,y1)
    loss = y.sum()
    loss1 = y1.sum()
    loss.backward()
    loss1.backward()
    # print('a',a)
    # print('y',y,'\ny1',y1)
    print(s.grad, s1.grad)
    # print(a.grad,a1.grad)
    assert all(((s.grad - s1.grad).abs()<1e-5).flatten()),'error'
    assert all(((a.grad-a1.grad).abs()<1e-5).flatten()),'error'

    s = torch.ones((2,1,1,1),requires_grad=True)
    x = torch.rand((2,3,2,2))
    s0 = bsq_ext.weiquan_init(x, s)
    x = torch.sigmoid(x) - 0.5
    s1 = ((x.view(2,-1).max(1)[0]- x.view(2,-1).min(1)[0])*0.9).reshape(s.shape)
    print(s1, s0)
    assert all(((s1-s0).abs()<1e-5).flatten()), 'error'

