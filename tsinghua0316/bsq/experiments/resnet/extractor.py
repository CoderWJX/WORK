import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from bsq import quan


class MyExtractor(object):
    def __init__(self, model, model_type):
        self.name2module = defaultdict()
        self.module2name = defaultdict()
        self.handles_dict = defaultdict()
        self.gpu_dict = defaultdict(dict)
        self.model = model
        for name, layer in model.named_modules():
            if isinstance(layer, model_type):
                self.name2module[name] = layer
                self.module2name[layer] = name

    def myhook(self, module, input, output):
        self.gpu_dict[str(input[0].device)][self.module2name[module]] = input[0].detach()

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
