import torch
import random
import numpy as np


def init_seed(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def gen_param_group(model, weight_name_list, quant_name_list):
    # 分组优化，量化参数不进行weight_decay
    all_params = model.parameters()
    quant_params = []
    weight_params = []
    quant_a_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        if any([pname.endswith(k) for k in quant_name_list]):
            quant_params += [p]
            # print('quant_params',pname)
        elif (any([name in pname for name in weight_name_list]) and 'weight' in pname):
            weight_params += [p]
            # print('weight_params',pname)
        # else:
            # print('other_params',pname)
    params_id = list(map(id, weight_params)) + list(map(id, quant_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    return weight_params, quant_params, other_params
