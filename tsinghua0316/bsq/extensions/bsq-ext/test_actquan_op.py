import torch
import bsq_ext

from torch import autograd

class QuanFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, neg, pos):
        y,mask = bsq_ext.actquan_forward(x,s,neg,pos)
        ctx.save_for_backward(y,x,s,mask)
        ctx.neg = neg
        ctx.pos = pos
        return y
    
    @staticmethod
    def backward(ctx, grad_from_top):
        y,x,s,mask = ctx.saved_tensors
        grad_s = bsq_ext.actquan_backward(x,s,y,mask,ctx.neg,ctx.pos)
        return grad_from_top, grad_s, None, None

class QuanFunc2(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, neg, pos):
        x_bar = x / s
        mask = (x_bar < neg - 0.5).float() + (x_bar > pos+0.5).float()
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
    # a = t.FloatTensor([0.1,-0.2,0.9,0.6]).reshape(2,2,1,1)
    a = torch.rand((2,2,1,1))
    a.requires_grad_(True)
    a1 = a.detach().clone()
    a1.requires_grad_(True)
    s = torch.ones((1,),requires_grad=True)
    s1 = torch.ones((1,),requires_grad=True)
    neg = 0
    pos = 6
    y = QuanFunc.apply(a,0.1*s,neg,pos)
    y1 = QuanFunc2.apply(a1,0.1*s1,neg,pos)
    loss = y.sum()
    loss1 = y1.sum()
    loss.backward()
    loss1.backward()
    # print('a',a)
    # print('y',y,'\ny1',y1)
    print(s.grad, s1.grad)
    assert all(torch.abs(s.grad - s1.grad)<1e-5),'error'

    s = torch.ones((1),requires_grad=True)
    x = torch.rand((2,2,1,1))
    s = bsq_ext.actquan_init(x, s, 6, True)
    print(x.max().item()*0.9/6, s.item())
    assert abs(x.max().item()*0.9/6 - s.item()) <= 1e-5, 'error'

    s = torch.ones((1),requires_grad=True)
    x = torch.rand((2,2,1,1))
    s = bsq_ext.actquan_init(x, s, 6, False)
    print(x.max().item()*0.9/6 * 0.1 + 0.9, s.item())
    assert abs(x.max().item()*0.9/6 * 0.1 + 0.9 - s.item())<= 1e-5, 'error'
