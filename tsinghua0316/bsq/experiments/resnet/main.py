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
        if not args.pretrained_path:
            logger.info('error, 本程序支持finetuning，请指定baseline模型的位置')
            exit(-1)
        modules_to_replace = quantized_qat_converter.find_modules_to_quantize(
            model, args.quan)
        model = quantized_qat_converter.replace_module_by_names(
            model, modules_to_replace)
        logger.info('Inserted qat quantizers into the original model')
        model = model.to(args.device.type)
        for i, (inputs, _) in enumerate(train_loader, 1):
            with torch.no_grad():
                _ = model(inputs.to(args.device.type))
            if i >= args.quan.ptq_batches:
                break
        logger.info('initialized activation quantization parameters')
    elif args.task_name == 'baseline':
        logger.info('train base model')
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
        criterion = torch.nn.CrossEntropyLoss().to(args.device.type)
        top1, top5, loss,all_size = process.validate(val_loader, model, criterion, -1, pymonitor, args, logger)
        util.save_checkpoint(-1, args.arch, model, {'top1': top1, 'top5': top5}, False, args.name, log_dir)

        # 新增转为onnx的模型
        logger.info('changing model to onnx formart')

        onnx_file_dst = os.path.join(log_dir, args.name + '_model.onnx')

        sample_input = {
            'inputs': torch.randn((32, 3, 224, 224)).cuda()
        }
        model.eval()
        with torch.no_grad():
            output = model(sample_input['inputs'])
        sample_output = {
            'outputs': output.detach()
        }
        # torch.onnx.export(model, tuple(sample_input.values()), onnx_file_dst,
        #                   input_names=sample_input.keys(),
        #                   output_names=sample_output.keys(),
        #                   export_params=True,
        #                   opset_version=11,
        #                   do_constant_folding=True,
        #                   )

        #添加转成int8保存的代码
        model_dict = model.module.state_dict() if isinstance(
            model, torch.nn.DataParallel) else model.state_dict()
        int8_file_dst = os.path.join(log_dir, args.name + '_int8model.pth')
        for k, v in model_dict.items():
            if 'weight' in k:
                model_dict[k] = v.cpu().to(dtype=torch.int8)
            elif 'scale' in k or 'zeropoint' in k:
                model_dict[k] = v.cpu().to(dtype=torch.float32)
        
        torch.save(model_dict, int8_file_dst)

        logger.info('Program completed successfully ... exiting ...')
        return
    else:
        logger.info('unknown task_name')
        exit(-1)

    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, lean=args.resume.lean)
        logger.info(f'resum model from {args.resume.path}')

    model = model.to(args.device.type)
    if args.device.gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device.gpu)
    criterion = torch.nn.CrossEntropyLoss().to(args.device.type)

    weight_params, quant_params, other_params = util.gen_param_group(
        model, ['conv', 'fc', 'linear'], ['quan_w_fn.s', 'quan_a_fn.s'])
    optimizer = torch.optim.SGD([
        {'params': weight_params, 'lr': args.optimizer.weight.learning_rate,
            'momentum': args.optimizer.weight.momentum, 'weight_decay': args.optimizer.weight.weight_decay},
        {'params': other_params, 'lr': args.optimizer.other.learning_rate,
         'momentum': args.optimizer.other.momentum, 'weight_decay': args.optimizer.other.weight_decay},
    ])
    quan_optimizer = None
    if args.task_name == 'quant':
        quan_optimizer = torch.optim.SGD([
            {'params': quant_params, 'lr': args.optimizer.quant.learning_rate,
                'momentum': args.optimizer.quant.momentum, 'weight_decay': args.optimizer.quant.weight_decay},
        ])
    lr_scheduler = MultiStepLR(
        optimizer, args.lr_scheduler.milestones, args.lr_scheduler.gamma)

    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(
        args.log.num_best_scores, logger)

    if args.eval:
        top1, top5, loss, all_size = process.validate(
            val_loader, model, criterion, -1, pymonitor, args, logger)
    else:  # training
        if args.debug:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, loss, all_size = process.validate(
                val_loader, model, criterion, -1, pymonitor, args, logger)
            util.save_checkpoint(-1, args.arch, model,
                                 {'top1': top1, 'top5': top5}, False, args.name, log_dir)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss, all_size = process.train(train_loader, model, criterion, optimizer,
                                                             epoch, pymonitor, args, logger, quan_optimizer=quan_optimizer)
            v_top1, v_top5, v_loss, all_size = process.validate(
                val_loader, model, criterion, epoch, pymonitor, args, logger)
            lr_scheduler.step()
            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {
                                 'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        top1, top5, loss, all_size = process.validate(
            val_loader, model, criterion, -1, pymonitor, args, logger)
    logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    args = util.get_config(default_file=Path.cwd()/'config.yaml')
    main(args)
