import math
import operator
import time

import torch

from bsq.util import AverageMeter, DataPrefetcher



__all__ = ['train', 'validate', 'PerformanceScoreboard']





def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, monitor, args, logger=None,quan_optimizer=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()

    rank = args.device.type
    if rank == 'cuda':
        train_loader = DataPrefetcher(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs = inputs.to(rank)
        targets = targets.to(rank)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        if quan_optimizer:
            quan_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if quan_optimizer:
            quan_optimizer.step()


        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if (batch_idx + 1) % args.log.print_freq == 0:
            monitor.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'weight_LR': optimizer.param_groups[0]['lr'],
                    'quant_LR': -1 if not quan_optimizer else quan_optimizer.param_groups[0]['lr'],
                    'other_LR':optimizer.param_groups[1]['lr']
                })
        
    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg, total_sample


def validate(data_loader, model, criterion, epoch, monitor, args, logger=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    if logger:
        logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    rank = args.device.type
    if rank == 'cuda':
        data_loader = DataPrefetcher(data_loader)
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            if (batch_idx + 1) % args.log.print_freq == 0:
                monitor.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    if logger:
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg, total_sample


class PerformanceScoreboard:
    def __init__(self, num_best_scores, logger):
        self.board = list()
        self.num_best_scores = num_best_scores
        self.logger = logger

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            if self.logger:
                self.logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
