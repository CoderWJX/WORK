from .quantized_module import RescaleReLULayer, QuanConv2dRescaleReLU, QuanLinearRescaleReLU

from .quantized_qat_module import QuanConv2d, QuanLinear

from collections import defaultdict

import bsq_ext

import torch


def gen_graph(model, input_size, default_layer=[torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear], branch_str='downsample', branch_type='bottle_neck'):
    module2name_dict = {}
    for n, m in model.named_modules():
        if any([isinstance(m, layer_type) for layer_type in default_layer]):
            module2name_dict[m] = n
    module_list = []
    skip_branch_list = []
    next_dict = defaultdict(list)
    pre_dict = defaultdict(list)
    skip_flag = 0
    if branch_type == 'bottle_neck':
        skip_length = -7
    elif branch_type == 'basic_neck':
        skip_length = -6

    def forward_hook(module, input, output):
        nonlocal skip_flag
        name = module2name_dict[module]
        if not module_list:
            module_list.append(module)
        else:
            if skip_flag:
                next_dict[skip_branch_list[-1]].append(module)
                pre_dict[module].append(skip_branch_list[-1])
                if branch_str in name:
                    skip_branch_list.append(module)
                else:
                    next_dict[module_list[-1]].append(module)
                    pre_dict[module].append(module_list[-1])
                    module_list.append(module)
                    skip_flag = 0
            else:
                if branch_str in name:
                    skip_branch_list.append(module)
                    next_dict[module_list[skip_length]].append(module)
                    pre_dict[module].append(module_list[skip_length])
                    skip_flag = 1
                else:
                    next_dict[module_list[-1]].append(module)
                    pre_dict[module].append(module_list[-1])
                    module_list.append(module)
        return
    handle_list = []
    for name, m in model.named_modules():
        if any([isinstance(m, layer_type) for layer_type in default_layer]):
            handle_list.append(m.register_forward_hook(forward_hook))
    x = torch.rand(input_size)
    model.eval()
    with torch.no_grad():
        model(x)
    for handle in handle_list:
        handle.remove()
    return next_dict, pre_dict, module2name_dict


def find_modules_to_convert(model):
    replaced_modules = dict()
    default_layers = [QuanConv2d, torch.nn.BatchNorm2d,
                      QuanLinear, torch.nn.Identity]
    branch_str = 'downsample'
    next_dict, pre_dict, module2name_dict = gen_graph(
        model, (1, 3, 224, 224), default_layer=default_layers, branch_str=branch_str, branch_type='bottle_neck')
    conv1_flag = 1
    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d):
            # 先处理第一个部分replace_conv部分
            if conv1_flag:
                replace_conv = RescaleReLULayer(
                    module.quan_a_fn.thd_pos/module.quan_a_fn.s.data * torch.ones(1, 3, 1, 1), None, out_type='int8')
                conv1_flag = 0
            elif branch_str in name:
                parent = pre_dict[module][0]
                brother = next_dict[parent][0] if next_dict[parent][-1] == module else next_dict[parent][-1]
                if isinstance(brother, torch.nn.Identity):
                    raise NotImplementedError
                else:
                    brother_sx = brother.quan_a_fn.s.data/brother.quan_a_fn.thd_pos
                this_sx = module.quan_a_fn.s.data/module.quan_a_fn.thd_pos
                replace_conv = RescaleReLULayer(brother_sx / this_sx * torch.ones(
                    1, module.weight.shape[1], 1, 1), None, out_type='uint8')
            # 如果这一层是由前两个节点的值相加得到，则表明这一层要先做处理，因为上一层的输出用的是int16,相加完之后是要relu并且又回到uint8的
            elif len(pre_dict[module]) == 2:
                replace_conv = RescaleReLULayer(torch.ones(
                    1, module.weight.shape[1], 1, 1), None, out_type='uint8')
            elif len(next_dict[pre_dict[module][0]]) == 2:
                replace_conv = RescaleReLULayer(torch.ones(
                    1, module.weight.shape[1], 1, 1), None, out_type='uint8')
            else:
                replace_conv = torch.nn.Identity()

            replaced_modules[module2name_dict[module]] = replace_conv

            # 再处理replace_bn
            weight, _ = bsq_ext.weiquan_forward(
                module.weight, module.quan_w_fn.s, module.quan_w_fn.intervals, module.quan_w_fn.thd_neg, module.quan_w_fn.thd_pos)
            this_sw = module.quan_w_fn.s.data/module.quan_w_fn.intervals
            weight /= this_sw
            this_sw = this_sw.view(-1)
            this_sx = module.quan_a_fn.s.data/module.quan_a_fn.thd_pos

            next_bn = next_dict[module][0]
            gamma = next_bn.weight.data
            beta = next_bn.bias.data
            mu = next_bn.running_mean
            sigma = torch.sqrt(next_bn.running_var + 1e-5)

            # 关键在于下一层的激活s如何得到
            if len(next_dict[next_bn]) == 2:
                next_conv = next_dict[next_bn][0] if branch_str in module2name_dict[next_dict[next_bn]
                                                                                    [-1]] else next_dict[next_bn][-1]
                if isinstance(next_conv, torch.nn.Identity):
                    raise NotImplementedError
                elif branch_str not in module2name_dict[next_conv]:
                    next_sx = next_conv.quan_a_fn.s.data/next_conv.quan_a_fn.thd_pos
                else:
                    raise NotImplementedError
                out_type = 'int16'
            else:
                next_conv = next_dict[next_bn][0]
                next_sx = next_conv.quan_a_fn.s.data/next_conv.quan_a_fn.thd_pos
                if len(pre_dict[next_conv]) == 2:
                    out_type = 'int16'
                else:
                    out_type = 'uint8'

            scale = gamma * this_sw * this_sx/(sigma * next_sx)
            zeropoint = (beta - gamma * mu/sigma)/next_sx
            replace_bn = QuanConv2dRescaleReLU(
                module, scale.reshape(1, -1, 1, 1), zeropoint.reshape(1, -1, 1, 1), out_type=out_type)
            replace_bn.conv.weight.data = weight
            replaced_modules[module2name_dict[next_bn]] = replace_bn
        elif isinstance(module, QuanLinear):
            weight, _ = bsq_ext.weiquan_forward(
                module.weight, module.quan_w_fn.s, module.quan_w_fn.intervals, module.quan_w_fn.thd_neg, module.quan_w_fn.thd_pos)
            this_sw = module.quan_w_fn.s.data/module.quan_w_fn.intervals
            weight /= this_sw
            this_sw = this_sw.view(-1)
            scale = this_sw
            zeropoint = module.bias.data
            replace_linear = QuanLinearRescaleReLU(
                module, scale.reshape(1, -1), zeropoint.reshape(1, -1))
            replace_linear.fc.weight.data = weight
            replaced_modules[module2name_dict[module]] = replace_linear
        elif isinstance(module, torch.nn.Identity):
            parent = pre_dict[module][0]
            brother = next_dict[parent][0] if next_dict[parent][-1] == module else next_dict[parent][-1]
            if isinstance(brother, torch.nn.Identity):
                raise NotImplementedError
            else:
                brother_sx = brother.quan_a_fn.s.data/brother.quan_a_fn.thd_pos
            next_conv = next_dict[module][0]
            this_sx = next_conv.quan_a_fn.s.data/next_conv.quan_a_fn.thd_pos
            replace_identity = RescaleReLULayer(brother_sx / this_sx * torch.ones(
                1, next_conv.weight.shape[1], 1, 1), None, out_type='uint16')
            replaced_modules[module2name_dict[module]] = replace_identity
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
