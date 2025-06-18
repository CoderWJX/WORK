import torch
from .quantized_module import QuantizedScalarMLPFunction
from .quant_qat_module import QuantScalarMLPFunction


def gen_graph(model, default_layer):
    module2name_dict = {}
    for name, module in model.named_modules():
        if any([isinstance(module, layer_type) for layer_type in default_layer]):
            module2name_dict[module] = name
    return module2name_dict

def find_modules_to_convert(model):
    replaced_modules = dict()
    default_layers = [QuantScalarMLPFunction]
    module2name_dict = gen_graph(model, default_layer=default_layers)
    quantzed_para_num = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantScalarMLPFunction):
            quantzed_para_num += module.w_0.numel()
            wei_scale = module.quan_w_fn.s.data
            quan_weight = torch.round(torch.clamp((module.w_0 * module.para) / wei_scale, module.quan_w_fn.thd_neg, module.quan_w_fn.thd_pos))
            module.w_0 = quan_weight
            act_scale = module.quan_a_fn.s.data
            replace_ScalarMLPFunction = QuantizedScalarMLPFunction(module, act_scale, wei_scale, name)
            replaced_modules[module2name_dict[module]] = replace_ScalarMLPFunction
    print("quantzed_para_num: %d" % quantzed_para_num)
    return replaced_modules

def replace_module_to_inference(model, modules_to_replace):
    need_replaced = {}
    for full_name, module in model.named_modules():
        if full_name in modules_to_replace.keys():
            need_replaced[module] = full_name

    def helper(child: torch.nn.Module):
        for name, module in child.named_children():
            if module in need_replaced.keys():
                child.add_module(name, modules_to_replace.pop(need_replaced[module]))
            else:
                helper(module)

    helper(model)
    return model