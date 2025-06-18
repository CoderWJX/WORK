from bsq.quan.quantizer import WeightQuan, ActQuan, IdentityQuan
from bsq.quan.quantizer.bsq import ActQuanFunc
import torch
from bsq.quan.quantized_qat_module import QuanConv2d, QuanLinear, QuanEmbedding
import torch.nn.functional as F

class QuantizedLinear(torch.nn.Linear):
    def __init__(self, m: QuanLinear):
        assert type(m) == QuanLinear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.w_s = 1
        if isinstance(m.quan_w_fn, torch.nn.Identity) or isinstance(m.quan_w_fn, IdentityQuan):
            self.weight = m.weight
        elif isinstance(m.quan_w_fn, WeightQuan):
            self.weight = torch.nn.Parameter(m.quan_w_fn(m.weight.detach())*m.quan_w_fn.intervals/m.quan_w_fn.s)
            self.w_s = torch.nn.Parameter(m.quan_w_fn.s/m.quan_w_fn.intervals)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        
        self.quan_a_fn = m.quan_a_fn
#         self.quan_w_fn = m.quan_w_fn
#         self.weight.data = m.weight.data
        self.register_buffer('t', m.t.detach())

    def forward(self, x):
        x = self.quan_a_fn(x)
#         weight = self.quan_w_fn(self.weight)
#         return F.linear(x * self.t, weight, self.bias)
        return F.linear(x * self.t, self.weight*self.w_s, self.bias)

    
class QuantizedEmbedding(torch.nn.Embedding):
    def __init__(self, m: QuanEmbedding):
        assert type(m) == QuanEmbedding
        super().__init__(m.num_embeddings, m.embedding_dim)
        self.w_s = 1
        if isinstance(m.quan_w_fn, torch.nn.Identity) or isinstance(m.quan_w_fn, IdentityQuan):
            self.weight = m.weight
        elif isinstance(m.quan_w_fn, WeightQuan):
            self.weight = torch.nn.Parameter((m.quan_w_fn(m.weight.detach().T)*m.quan_w_fn.intervals/m.quan_w_fn.s).T)
            self.w_s = torch.nn.Parameter(m.quan_w_fn.s/m.quan_w_fn.intervals)
        self.register_buffer('t', m.t.detach())
        self.quan_a_fn = m.quan_a_fn
#         self.weight.data = m.weight.data
        
    def forward(self, x):
        return F.embedding(x, (self.weight.T * self.w_s).T * self.t)
#         return F.embedding(x, self.quan_w_fn(self.weight.T).T * self.t)
    

def find_modules_to_convert(model):
    replaced_modules = dict()
    for full_name, m in model.named_modules():
        if isinstance(m, QuanLinear):
            replaced_modules[full_name] = QuantizedLinear(m)
        elif isinstance(m, QuanEmbedding):
            replaced_modules[full_name] = QuantizedEmbedding(m)
    return replaced_modules

def replace_module_to_inference(model, modules_to_replace):
    need_replaced = {}
    for full_name, m in model.named_modules():
        if full_name in modules_to_replace.keys():
            need_replaced[m] = full_name

    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            if c in need_replaced.keys():
                child.add_module(n, modules_to_replace.pop(need_replaced[c]))
            else:
                helper(c)

    helper(model)
    return model
