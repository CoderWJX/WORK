import torch

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np 


class MyExtractor(object):
    def __init__(self, model, model_type):
        self.name2module = defaultdict()
        self.module2name = defaultdict()
        self.handles_dict = defaultdict()
        self.gpu_dict = defaultdict(dict)
        self.model = model
        for name, layer in model.named_modules():
            print(name)
            if isinstance(layer, model_type):
                print('ok')
                self.name2module[name] = layer
                self.module2name[layer] = name

    def myhook(self, module, input, output):
        this_dict = self.gpu_dict[str(input[0].device)]
        if self.module2name[module] not in this_dict:
            this_dict[self.module2name[module]] = input[0].max(0)[0].detach().cpu()
        else:
            this_dict[self.module2name[module]] = this_dict[self.module2name[module]] * 0.1 + input[0].max(0)[0].detach().cpu() * 0.9

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


def ptq_act_equalization(model, train_loader, device, iters=1000):
    model = model.to(device)
    extractor = MyExtractor(model, torch.nn.Linear)
    extractor.begin_extractor()
    for i, ((user,item), _) in enumerate(train_loader, 1):
        with torch.no_grad():
            _ = model(user.to(device), item.to(device))
        if i >= 1000:
            break
    extractor.end_extractor()
    print('done!')
    return extractor.gpu_dict