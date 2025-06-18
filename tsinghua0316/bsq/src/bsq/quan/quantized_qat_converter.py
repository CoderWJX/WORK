import logging

from .quantized_qat_module import QuanModuleMapping
from .quantizer import IdentityQuan, WeightQuan, ActQuan, LsqQuan
import torch


def quantizer(default_cfg, this_cfg=None, ptq_batches=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'weightquan':
        q = WeightQuan
    elif target_cfg['mode'] == 'actquan':
        q = ActQuan
        target_cfg['ptq_batches'] = ptq_batches
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
        if 't_gamma' in target_cfg:
            target_cfg.pop('t_gamma')
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
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
                                        ptq_batches=quan_scheduler.ptq_batches),
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
