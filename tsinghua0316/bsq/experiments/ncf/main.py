import logging
from pathlib import Path

import torch
import yaml

import process
from bsq.quan import quantized_converter, quantized_qat_converter
from bsq.quan.quantizer.bsq import ActQuanFunc
from bsq import util
from bsq.model import create_model
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR

import os

import time
import evaluate_ncf
import numpy as np

from bsq.util import DataPrefetcher

from bsq.util import AverageMeter

import convert_tool
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
    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num ,item_num, train_mat = util.ncf_dataloader.load_all(args.dataloader)

    # construct the train and test datasets
    train_dataset = util.ncf_dataloader.NCFData(
        train_data, item_num, train_mat, args.dataloader.num_ng, True)
    test_dataset = util.ncf_dataloader.NCFData(
        test_data, item_num, train_mat, 0, False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.dataloader.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.dataloader.test_num_ng+1, shuffle=False, num_workers=4)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    print("hwl debug:", args)
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
        
        GMF_model_dict = None
        MLP_model_dict = None
        if 'GMF_path' in args.__dict__ and args.GMF_path and os.path.exists(args.GMF_path):
            GMF_model_dict = torch.load(args.GMF_path, map_location='cpu')
        if 'MLP_path' in args.__dict__ and args.MLP_path and os.path.exists(args.MLP_path):
            MLP_model_dict = torch.load(args.MLP_path, map_location='cpu')
        if GMF_model_dict and MLP_model_dict:
            net_dict = model.state_dict()
            for name in net_dict.keys():
                if 'predict_layer.weight' in name:
                    net_dict[name] =0.5 * torch.cat((GMF_model_dict[name], MLP_model_dict[name]), -1)
                elif 'predict_layer.bias' in name:
                    net_dict[name] =0.5 * (GMF_model_dict[name] + MLP_model_dict[name])
                elif name in MLP_model_dict.keys():
                    net_dict[name] = MLP_model_dict[name]
                    print('mlp', name)
                elif name in GMF_model_dict.keys():
                    net_dict[name] = GMF_model_dict[name]
                    print('gmf', name)
            model.load_state_dict(net_dict)
            model = model.to(args.device.type)
        else:
            model = model.to(args.device.type)
            for i, ((user,item), _) in enumerate(train_loader, 1):
                with torch.no_grad():
                    _ = model(user.to(args.device.type), item.to(args.device.type))
                if i >= 2 * args.quan.ptq_batches:
                    break
        logger.info('initialized activation quantization parameters')
        logger.info('%s\n' % model)
    elif args.task_name == 'baseline':
        logger.info('baseline!')
        GMF_model_dict = None
        MLP_model_dict = None
        if 'GMF_path' in args.__dict__ and args.GMF_path and os.path.exists(args.GMF_path):
            GMF_model_dict = torch.load(args.GMF_path, map_location='cpu')
        if 'MLP_path' in args.__dict__ and args.MLP_path and os.path.exists(args.MLP_path):
            MLP_model_dict = torch.load(args.MLP_path, map_location='cpu')
        if GMF_model_dict and MLP_model_dict:
            net_dict = model.state_dict()
            for name in net_dict.keys():
                if 'predict_layer.weight' in name:
                    net_dict[name] =0.5 * torch.cat((GMF_model_dict[name], MLP_model_dict[name]), -1)
                elif 'predict_layer.bias' in name:
                    net_dict[name] =0.5 * (GMF_model_dict[name] + MLP_model_dict[name])
                elif name in MLP_model_dict.keys():
                    net_dict[name] = MLP_model_dict[name]
                    print('mlp', name)
                elif name in GMF_model_dict.keys():
                    net_dict[name] = GMF_model_dict[name]
                    print('gmf', name)
            model.load_state_dict(net_dict)
            model = model.to(args.device.type)
            
    elif args.task_name == 'inference':
        modules_to_replace = quantized_qat_converter.find_modules_to_quantize(
            model, args.quan)
        model = quantized_qat_converter.replace_module_by_names(
            model, modules_to_replace)
        
        print("hwl debug:", args.quant_path)
        model.load_state_dict(torch.load(args.quant_path,map_location='cpu'))
        modules_to_convert = convert_tool.find_modules_to_convert(model)
        model = convert_tool.replace_module_to_inference(model, modules_to_convert)
        model_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        first_layer = None
        for name, m in model.named_modules():
            if 'MLP_layers.1' in name:
                first_layer = m
                break
        for name, m in model.named_modules():
            if 'embed_item_MLP' in name or 'embed_user_MLP' in name:
                m.weight.data = ActQuanFunc.apply(m.weight.data,first_layer.a_s,first_layer.a_low, first_layer.a_high)/first_layer.a_s
                m.w_s = torch.nn.Parameter(first_layer.a_s)
        first_layer.is_quan_a = False
        model = model.to(args.device.type)
        model.eval()
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device.type)
        if args.extract:
            myextractor = MyExtractor(model)
        _, HR, NDCG = evaluate_ncf.metrics(model, test_loader, args.dataloader.top_k, criterion, myextractor)
        if args.extract:
            input_dict = myextractor.get_extractor_dict()
            input_filename = os.path.join(log_dir, args.name + 'extract_input.pth')
            torch.save(input_dict, input_filename)
        logger.info("loss:{:.4f}\tHR: {:.4f}\tNDCG: {:.4f}".format(-1,np.mean(HR), np.mean(NDCG)))
        model_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        
        
        filename_best_pth = 'best.pth' if args.name is None else args.name + '_best.pth'
        filepath_best_pth = os.path.join(log_dir, filename_best_pth)
        torch.save(model_dict, filepath_best_pth)
        need_int8 = []
        for name, param in model_dict.items():
            if name.endswith('.w_s'):
                need_int8.append(name.replace('w_s', 'weight'))
        for name, param in model_dict.items():
            if name in need_int8:
                model_dict[name] = param.to(torch.int8)
        filename_best_pth = 'best_int8.pth' if args.name is None else args.name + '_best_int8.pth'
        filepath_best_pth = os.path.join(log_dir, filename_best_pth)
        torch.save(model_dict, filepath_best_pth)
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
    criterion = torch.nn.BCEWithLogitsLoss().to(args.device.type)

    weight_params, quant_params, other_params = util.gen_param_group(
        model, ['embed', 'MLP', 'predict'], ['quan_w_fn.s', 'quan_a_fn.s'])
    if args.optimizer.type == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': weight_params, 'lr': args.optimizer.weight.learning_rate,
                'momentum': args.optimizer.weight.momentum, 'weight_decay': args.optimizer.weight.weight_decay},
            {'params': other_params, 'lr': args.optimizer.other.learning_rate,
             'momentum': args.optimizer.other.momentum, 'weight_decay': args.optimizer.other.weight_decay},
        ])
    elif args.optimizer.type == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': weight_params, 'lr': args.optimizer.weight.learning_rate,'initial_lr': args.optimizer.weight.learning_rate,
                'weight_decay': args.optimizer.weight.weight_decay},
           {'params': other_params, 'lr': args.optimizer.other.learning_rate,'initial_lr': args.optimizer.weight.learning_rate,
            'weight_decay': args.optimizer.other.weight_decay},
        ])
    quan_optimizer = None
    if args.task_name == 'quant':
        quan_optimizer = torch.optim.SGD([
            {'params': quant_params, 'lr': args.optimizer.quant.learning_rate,
                'momentum': args.optimizer.quant.momentum, 'weight_decay': args.optimizer.quant.weight_decay},
        ])
    lr_scheduler = None
    if args.lr_scheduler.type == 'multistep':
        lr_scheduler = MultiStepLR(
            optimizer, args.lr_scheduler.milestones, args.lr_scheduler.gamma)
    elif args.lr_scheduler.type == 'step':
        lr_scheduler = StepLR(optimizer, args.lr_scheduler.step_size, gamma=args.lr_scheduler.gamma, last_epoch=args.lr_scheduler.last_epoch)
    elif args.lr_scheduler.type == 'exp':
        lr_scheduler = ExponentialLR(optimizer, gamma=args.lr_scheduler.gamma, last_epoch=args.lr_scheduler.last_epoch)

    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(
        args.log.num_best_scores, logger)

    if args.eval:
        model.eval()
        _, HR, NDCG = evaluate_ncf.metrics(model, test_loader, args.dataloader.top_k, criterion)
        logger.info("loss:{:.4f}\tHR: {:.4f}\tNDCG: {:.4f}".format(-1,np.mean(HR), np.mean(NDCG)))
    else:  # training
        if args.debug:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            model.eval()
            _, HR, NDCG = evaluate_ncf.metrics(model, test_loader, args.dataloader.top_k, criterion)
            util.save_checkpoint(-1, args.arch, model,
                                 {'hr': HR, 'ndcg': NDCG}, True, args.name, log_dir)
            perf_scoreboard.update(HR, NDCG, start_epoch - 1)
            logger.info("loss:{:.4f}\tHR: {:.4f}\tNDCG: {:.4f}".format(-1,np.mean(HR), np.mean(NDCG)))
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            model.train() # Enable dropout (if have).
            start_time = time.time()
            avg_loss = AverageMeter()
#             train_loader = DataPrefetcher(train_loader)

            for (user, item), label in train_loader:
                user = user.cpu()
                item = item.cpu()
                label = label.float().cpu()

                prediction = model(user, item)
                loss = criterion(prediction, label.reshape(-1))
                optimizer.zero_grad()
                if quan_optimizer:
                    quan_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if quan_optimizer:
                    quan_optimizer.step()
                avg_loss.update(loss.item(), prediction.shape[0])
            lr_scheduler.step()
                
            model.eval()
            test_loss, HR, NDCG = evaluate_ncf.metrics(model, test_loader, args.dataloader.top_k, criterion)
            perf_scoreboard.update(HR, NDCG, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model,
                                 {'hr': HR, 'ndcg': NDCG}, is_best, args.name, log_dir)
            elapsed_time = time.time() - start_time
            logger.info("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            logger.info("loss:{:.4f}\ttest_loss:{:.4f}\tHR: {:.4f}\tNDCG: {:.4f}\tweight_lr: {:.7f}\tinitial_lr:{:.7f}".format(avg_loss.avg,test_loss,np.mean(HR), np.mean(NDCG), optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['initial_lr'] if 'initial_lr' in optimizer.param_groups[0].keys() else 0))

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        _, HR, NDCG = evaluate_ncf.metrics(model, test_loader, args.dataloader.top_k, criterion)
        logger.info("loss:{:.4f}\tHR: {:.4f}\tNDCG: {:.4f}".format(-1,np.mean(HR), np.mean(NDCG)))
    logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    args = util.get_config(default_file=Path.cwd()/'config.yaml')
    main(args)
