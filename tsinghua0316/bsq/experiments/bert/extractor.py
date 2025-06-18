import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from bsq import quan
from convert_tool import QuantizedLinear, QuantizedEmbedding
from bsq.quan.quantizer.bsq import ActQuanFunc



class MyExtractor(object):
    def __init__(self, model):
        self.name2module = defaultdict()
        self.module2name = defaultdict()
        self.handles_dict = defaultdict()
        self.gpu_dict = defaultdict(list)
        self.model = model
        for name, layer in model.named_modules():
            if isinstance(layer, QuantizedLinear) or isinstance(layer, QuantizedEmbedding):
                self.name2module[name] = layer
                self.module2name[layer] = name
                layer.full_name = name

    def myhook(self, module, input, output):
        need = module.quan_a_fn(input[0].detach())
        self.gpu_dict[str(input[0].device)][module.full_name] = need

    def begin_extractor(self):
        self.handles_dict = defaultdict()
        self.gpu_dict = defaultdict(dict)
        for name, layer in self.name2module.items():
            func = lambda *x: self.myhook(*x)
            self.handles_dict[name] = layer.register_forward_hook(func)

    def end_extractor(self):
        for name, handle in self.handles_dict.items():
            handle.remove()

    def get_extractor_dict(self):
        mask_dict = {}
        for name, value in self.gpu_dict.items():
            for k, v in value.items():
                mask_dict[k + '_in_' + name] = v
        return mask_dict
