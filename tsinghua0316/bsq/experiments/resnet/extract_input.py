import logging
from pathlib import Path

import torch
import yaml

import process
from bsq.quan import quantized_converter, quantized_qat_converter
from bsq import util
from bsq.model import create_model
from torch.optim.lr_scheduler import MultiStepLR

import os

from extractor import MyExtractor


def main(args):

    util.init_seed(args.dataloader.seed, args.dataloader.deterministic)

    script_dir = Path.cwd()

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(
        args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)
    pymonitor = util.ProgressMonitor(logger)
    if args.device.type == 'cpu' or not torch.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = torch.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        torch.cuda.set_device(args.device.gpu[0])

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    model = create_model(args)
    if args.task_name == 'quant':
        logger.info('此脚本用于抽取量化模型的中间层，不适用于qat量化')
        exit(-1)
    elif args.task_name == 'baseline':
        logger.info('此脚本用于抽取量化模型的中间层，不适用于训练baseline')
        exit(-1)
    elif args.task_name == 'inference':
        modules_to_replace = quantized_qat_converter.find_modules_to_quantize(
            model, args.quan)
        model = quantized_qat_converter.replace_module_by_names(
            model, modules_to_replace)
        model.load_state_dict(torch.load(args.quant_path, map_location='cpu'))

        logger.info('Inserted quantizers into the original model')

        modules_to_inference = quantized_converter.find_modules_to_convert(
            model)
        model = quantized_converter.replace_module_to_inference(
            model, modules_to_inference)
        logger.info('changed quantized model into inference model')

        util.save_checkpoint(-1, args.arch, model,
                             {'top1': 0, 'top5': 0}, False, args.name, log_dir)

        model = model.to(args.device.type)
        
        myext = MyExtractor(model, torch.nn.Conv2d)
        
        for inputs, _ in train_loader:
            myext.begin_extractor()
            model(inputs.to(args.device.type))
            myext.end_extractor()
            torch.save(myext.get_extractor_dict(), os.path.join(log_dir, args.name + '_inputs.pth.tgz'))
            break

        logger.info('Program completed successfully ... exiting ...')
        return
    else:
        logger.info('unknown task_name')
        exit(-1)


if __name__ == "__main__":
    args = util.get_config(default_file=Path.cwd()/'config.yaml')
    main(args)
