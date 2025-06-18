import torch 
from torch import autograd

from .quantizer import Quantizer
import bsq_ext


class ActQuanFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, neg, pos):
        y, mask = bsq_ext.actquan_forward(x, s, neg, pos)
        ctx.save_for_backward(y, x, s, mask)
        ctx.neg = neg
        ctx.pos = pos
        return y

    @staticmethod
    def backward(ctx, grad_from_top):
        y, x, s, mask = ctx.saved_tensors
        grad_s = bsq_ext.actquan_backward(x, s, y, mask, ctx.neg, ctx.pos)
        return grad_from_top, grad_s, None, None


class WeiQuanFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x, s, intervals, neg, pos):
        y, mask = bsq_ext.weiquan_forward(x, s, intervals, neg, pos)
        # print(mask)
        ctx.save_for_backward(y, x, s, mask)
        ctx.neg = neg
        ctx.pos = pos
        ctx.intervals = intervals
        return y

    @staticmethod
    def backward(ctx, grad_from_top):
        y, x, s, mask = ctx.saved_tensors
        grad_x, grad_s = bsq_ext.weiquan_backward(
            x, s, y, mask, ctx.intervals, ctx.neg, ctx.pos)
        return grad_from_top * grad_x, grad_s, None, None, None


class WeightQuan(Quantizer):

    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, **kargs):
        super().__init__(bit)

        assert not all_positive, "weight quantization cannot be all positive"
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
            self.intervals = 2**bit - 2
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = - 2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1
            self.intervals = 2**bit - 1

        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))
        self.t_gamma = kargs['t_gamma']

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if x.dim() == 4:
                self.s = torch.nn.Parameter(
                    torch.ones((x.shape[0], 1, 1, 1), device=x.device))
            elif x.dim() == 2:
                self.s = torch.nn.Parameter(
                    torch.ones((x.shape[0], 1), device=x.device))
            self.s.data = bsq_ext.weiquan_init(x, self.s.data)

    def forward(self, x):
        return WeiQuanFunc.apply(x, self.s, self.intervals, self.thd_neg, self.thd_pos)
        #return ActQuanFunc.apply(x, self.s/self.thd_pos, self.thd_neg, self.thd_pos)


class ActQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, ptq_batches=0, **kargs):
        super().__init__(bit)

        assert not per_channel, "activation quantization cannot be per_channel"
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.s = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('iterth', torch.zeros(1))
        self.ptq_batches = ptq_batches

    def forward(self, x):
        self.ptq_batches = 1000
        if self.iterth.item() < self.ptq_batches:
            self.s.data = bsq_ext.actquan_init(
                x, self.s, self.iterth.item() == 0)
            self.iterth += 1
            #if self.iterth.item() * 2 < self.ptq_batches:
            #return x

        x = ActQuanFunc.apply(x, self.s/self.thd_pos,
                              self.thd_neg, self.thd_pos)
        return x
