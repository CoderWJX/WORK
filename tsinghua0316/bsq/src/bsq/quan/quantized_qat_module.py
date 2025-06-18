import torch
import bsq_ext
from torch import autograd
import torch.nn.functional as F
from .quantizer import IdentityQuan, WeightQuan
import numpy as np

class QuanBaseFunc(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = bsq_ext.quant_base_forward(x)
        ctx.save_for_backward(y, x)
        return y

    @staticmethod
    def backward(ctx, grad_from_top):
        y, x = ctx.saved_tensors
        grad_x = bsq_ext.quant_base_backward(x, y)
        return grad_from_top * grad_x


class QuanBaseLayer(torch.nn.Module):
    def __init__(self):
        super(QuanBaseLayer, self).__init__()

    def forward(self, x):
        return QuanBaseFunc.apply(x)


class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.weight = torch.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.register_buffer('t', torch.ones(1))

        if  isinstance(quan_w_fn, WeightQuan):
            self.t.data = torch.ones(1) * self.weight.data.abs().max()*self.quan_w_fn.t_gamma
#             self.t.data = torch.ones(1) * np.percentile(self.weight.data.abs().numpy(), self.quan_w_fn.t_gamma * 100)
            self.weight.data = bsq_ext.quant_base_init(self.weight.data/self.t)
        else:
            print('identityquan')
        self.quan_w_fn.init_from(self.weight.data)

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return F.conv2d(quantized_act * self.t, quantized_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
    
class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)

        self.weight = torch.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.register_buffer('t', torch.ones(1))

        if isinstance(quan_w_fn, WeightQuan):
            self.t.data = torch.ones(1) * self.weight.data.abs().max()*self.quan_w_fn.t_gamma
#             self.t.data = torch.ones(1) * np.percentile(self.weight.data.abs().numpy(), self.quan_w_fn.t_gamma * 100)
            self.weight.data = bsq_ext.quant_base_init(self.weight.data/self.t)
        else:
            print('identityquan')
        self.quan_w_fn.init_from(self.weight.data)

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return F.linear(quantized_act * self.t, quantized_weight, self.bias)

    
class QuanEmbedding(torch.nn.Embedding):
    def __init__(self, m: torch.nn.Embedding, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Embedding
        super().__init__(m.num_embeddings, m.embedding_dim)
        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = torch.nn.Identity()
        self.register_buffer('t', torch.ones(1))
        if isinstance(quan_w_fn, WeightQuan):
            self.t.data = torch.ones(1) * self.weight.data.abs().max()*self.quan_w_fn.t_gamma
#             self.t.data = torch.ones(1) * np.percentile(self.weight.data.abs().numpy(), self.quan_w_fn.t_gamma * 100)
            self.weight.data = bsq_ext.quant_base_init(self.weight.data.T/self.t).T
        else:
            print('identityquan')
        self.quan_w_fn.init_from(self.weight.data.T)
        
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight.T).T
        quantized_act = self.quan_a_fn(x)
        return F.embedding(quantized_act, quantized_weight * self.t)
    

QuanModuleMapping = {
    torch.nn.Conv2d: QuanConv2d,
    torch.nn.Linear: QuanLinear,
    torch.nn.Embedding: QuanEmbedding
}
