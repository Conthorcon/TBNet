"""
File: train.py
Author: Truong-Khanh
Created: 2025-12-29

Description:
    Pass training args to the trainer and set training resources.

Main functions:
    get_parser: store training args
    main_worker: set up training variables, dataset and call trainer

Usage:
    python train.py --image_path data/test.jpg

Dependencies:
    - torch
    - loguru

Notes:

"""

import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from loguru import logger
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
import torch.cuda.amp as amp

from model import build_segmenter
import utils.config as config
# import wandb
from utils.dataloader import TrainDataset, TestDataset
from engine.engine import train, val
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)
from utils import viz



def get_parser():
    """
    Storing the training args.

    Args:
        None (None): None.

    Returns:
        dict: key and value of training args.

    Raises:
        ValueError: Nếu ảnh đầu vào rỗng hoặc không hợp lệ.
    """
    parser = argparse.ArgumentParser(
        description='ACUMEN')
    parser.add_argument('--config',
                        default='config/config.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg
    

def main_worker(args, shared_vars):
    """
    Docstring for main_worker
    
    :param args: (dict) stores the training args
    :param shared_vars: (list) stores the results of each training run for validation.
    """
    args.output_dir = os.path.join(args.map_save_path, args.exp_name)

    # build model
    print("building model")
    model, param_list = build_segmenter(args)

    # logger.info(model)
    model = model.cuda()

    # build optimizer & lr scheduler
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_list,
                                    lr=args.base_lr,
                                    weight_decay=args.weight_decay)

    if args.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer,
                                milestones=args.milestones,
                                gamma=args.lr_decay)
        
    if args.scaler == 'GradScaler':
        scaler = amp.GradScaler()

    # build dataset
    print('building dataset...')
    train_data = TrainDataset(image_root=args.train_root + '/ACUMEN/Imgs/',
                              gt_root=args.train_root + '/ACUMEN/GT/',
                              trainsize=args.input_size)
    
    val_data = TestDataset(image_root=args.val_root + '/Imgs/',
                              gt_root=args.val_root + '/GT/',
                              testsize=args.input_size)
    
    # build dataloader
    print('building dataloader...')
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      seed=args.manual_seed)


    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   drop_last=False)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 drop_last=False)


    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # train
        if epoch_log == 1:
            print("start training")
            
        train(train_loader, model, optimizer, scheduler, epoch_log, args)

        # evaluation & save
        # if epoch > args.epochs//2:
        val(val_loader, model, epoch_log, args, shared_vars)

        # update lr
        scheduler.step(epoch_log)


    # logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == "__main__":
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)

    # make shared dictionary among mp 
    shared_vars = dict()
    shared_vars['best_score'] = 0
    shared_vars['best_epoch'] = 0
    shared_vars['best_metric_dict'] = dict()

    main_worker(args, shared_vars)