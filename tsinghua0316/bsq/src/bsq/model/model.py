import logging

from .resnet_cifar import resnet20,resnet32,resnet44,resnet56,resnet110,resnet1202
import torch
# from torchvision import models
from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152

from .ncf import ncf8, ncf16, ncf32, ncf64, mlp8, mlp16, mlp32, mlp64, gmf8, gmf16, gmf32, gmf64

def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = resnet18()
        elif args.arch == 'resnet34':
            model = resnet34()
        elif args.arch == 'resnet50':
            model = resnet50()
        elif args.arch == 'resnet101':
            model = resnet101()
        elif args.arch == 'resnet152':
            model = resnet152()
    elif args.dataloader.dataset == 'cifar10':
        if args.arch == 'resnet20':
            model = resnet20()
        elif args.arch == 'resnet32':
            model = resnet32()
        elif args.arch == 'resnet44':
            model = resnet44()
        elif args.arch == 'resnet56':
            model = resnet56()
        elif args.arch == 'resnet110':
            model = resnet110()
        elif args.arch == 'resnet1202':
            model = resnet1202()
    elif args.dataloader.dataset == 'ml-1m':
        if args.arch == 'ncf8':
            model = ncf8(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'ncf16':
            model = ncf16(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'ncf32':
            model = ncf32(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'ncf64':
            model = ncf64(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'mlp8':
            model = mlp8(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'mlp16':
            model = mlp16(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'mlp32':
            model = mlp32(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'mlp64':
            model = mlp64(args.dataloader.user_num, args.dataloader.item_num)
        if args.arch == 'gmf8':
            model = gmf8(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'gmf16':
            model = gmf16(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'gmf32':
            model = gmf32(args.dataloader.user_num, args.dataloader.item_num)
        elif args.arch == 'gmf64':
            model = gmf64(args.dataloader.user_num, args.dataloader.item_num)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)
    print("hwl debug:", args.pretrained_path)
    if args.pretrained_path:
        model.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pretrained_path
    logger.info(msg)

    return model
