# from .quantization_qat_module import QuanModuleMapping
import torch
from allegro.nn._fc import ScalarMLPFunction
from .qt_util import quantizer
from .quant_qat_module import QuantScalarMLPFunction

def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = dict()
    for name, module in model.named_modules():
        quant_flag = False
        for key in QuantModuleNameLists:
            if name.find(key) != -1:
                quant_flag = True
        if type(module) in QuanModuleMapping.keys() and quant_flag:
            if name in quan_scheduler.excepts:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight,
                                        quan_scheduler.excepts[name].weight),
                    quan_a_fn=quantizer(quan_scheduler.act,
                                        quan_scheduler.excepts[name].act,
                                        ptq_batches=quan_scheduler.ptq_batches),
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight),
                    quan_a_fn=quantizer(quan_scheduler.act,
                                        ptq_batches=quan_scheduler.ptq_batches)
                )
        elif name in quan_scheduler.excepts:
            logging.warning(
                'Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
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


QuanModuleMapping = {
    ScalarMLPFunction: QuantScalarMLPFunction
}

QuantModuleNameLists= {
    "latents",
    "env_embed_mlps",
    # "linears",
    "final_latent",
    "edge_eng",
    # "env_linears"
}