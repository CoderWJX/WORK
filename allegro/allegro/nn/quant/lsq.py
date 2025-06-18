import torch as t

from .quantizer import Quantizer
import torch

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

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

        self.per_channel = per_channel
        # self.s = t.nn.Parameter(t.ones(1))
        self.s = t.nn.Parameter(t.tensor([0.008]))


    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                torch.tensor([x[i, :].detach().abs().max() for i in range(x.shape[0])], device=x.device).reshape(-1, 1) / self.thd_pos)
        else:
            self.s = t.nn.Parameter(x.detach().abs().max() * 2 / self.thd_pos)

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, torch.tensor(s_grad_scale))

        x = x / s_scale
        x = t.clamp(torch.nan_to_num(x), self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = torch.nan_to_num(x) * s_scale

        return x

