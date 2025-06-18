import time
import torch
import bsq_ext
import math

from torch import autograd

class BSQQuanFunc(autograd.Function):
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

class UNIQuanFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, neg, pos):
        x_bar = x / s
        mask = (x_bar < neg - 0.5).float() + (x_bar > pos+0.5).float()
        #print(mask)
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

def bsq_quantize(a, s, neg, pos):
    y = BSQQuanFunc.apply(a,s,1, neg,pos)

    loss = y.sum()
    loss.backward()

def uniform_quantize(a, s, neg, pos):
    y = UNIQuanFunc.apply(a,s,neg,pos)

    loss = y.sum()
    loss.backward()



if __name__ == '__main__':

    repeat_time=1000
    quant_type=1  # 1 means bsq, 0 means uniform quantization

    #a = torch.rand((512, 512,3,3))
    #a = torch.rand((512, 1024))
    neg = -128
    pos = 128

    # GPU
    for num_bs in [512, 1024]:
        for num_hs in [512, 1024, 2048]:
            #print("Test: ", a.shape)
            if quant_type == 0:
                #a = torch.rand((num_bs, num_hs, 3, 3)).cuda()
                a = torch.randn((num_bs, num_hs)).cuda()
                a.requires_grad_(True)
                #s_a = torch.ones((num_bs,1,1,1,),requires_grad=True).cuda()
                s_a = torch.ones((num_bs,1),requires_grad=True).cuda()

                start = time.time()
                for num in range(1,repeat_time):
                    uniform_quantize(a, s_a, neg, pos)
                end = time.time()
                print("[GPU] Uniform Quantize time cost:", "{:.4}".format((end-start)/repeat_time*1000), "  millisecond (ms)")

                del a
                del s_a
            else:
                #b = torch.rand((num_bs, num_hs, 3, 3)).cuda()
                b = torch.randn((num_bs, num_hs)).cuda()
                b.requires_grad_(True)
                #s_b = torch.ones((num_bs,1,1,1,),requires_grad=True).cuda()
                s_b = torch.ones((num_bs,1),requires_grad=True).cuda()

                start = time.time()
                for num in range(1,repeat_time):
                    bsq_quantize(b,  s_b, neg, pos)
                end = time.time()
                print("[GPU] BSQ Quantize time cost:", "{:.4}".format((end-start)/repeat_time*1000), " millisecond (ms)")

                del b
                del s_b


