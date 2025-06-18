import os

import numpy as np
import torch
import torchvision as tv
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

class DataPrefetcher():
    def __init__(self, loader):
        self.loaderiter = iter(loader)
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loaderiter)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, tuple) or isinstance(self.next_input, list):
                input = []
                for c in self.next_input:
                    input.append(c.cuda(non_blocking=True))
                self.next_input = input
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_input = self.next_input.float()
            self.next_target = self.next_target.cuda(non_blocking=True)
            

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def batch_size(self):
        return self.loader.batch_size

    def __iter__(self):
        count = 0
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input, target = self.next_input, self.next_target
            self.preload()
            count += 1
            yield input, target


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = torch.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = torch.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def load_data(args):
    cfg = args.dataloader
    if cfg.val_split < 0 or cfg.val_split >= 1:
        raise ValueError(
            'val_split should be in the range of [0, 1) but got %.3f' % cfg.val_split)

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    if cfg.dataset == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'train'), transform=train_transform)
        test_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'val'), transform=val_transform)

    elif cfg.dataset == 'cifar10':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, 4),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.CIFAR10(
            cfg.path, train=True, transform=train_transform, download=True)
        test_set = tv.datasets.CIFAR10(
            cfg.path, train=False, transform=val_transform, download=True)

    else:
        raise ValueError('load_data does not support dataset %s' % cfg.dataset)

    if cfg.val_split != 0:
        train_set, val_set = __balance_val_split(train_set, cfg.val_split)
    else:
        # In this case, use the test set for validation
        val_set = test_set

    train_sampler, test_sampler, val_sampler = None, None, None
    train_loader = DataLoader(
        train_set, cfg.batch_size, num_workers=cfg.workers, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(
        val_set, cfg.batch_size, num_workers=cfg.workers, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(
        test_set, cfg.batch_size, num_workers=cfg.workers, pin_memory=True, sampler=test_sampler)

    return train_loader, val_loader, test_loader
