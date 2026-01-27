"""
File: train.py
Author: Truong-Khanh
Created: 2026-1-06

Description:
    Evaluate model in datasets.

Main functions:
    get_parser: store training args
    main: set up training variables, dataset and call tester

Usage:
    python train.py --image_path data/test.jpg

Dependencies:
    - torch
    - loguru

Notes:

"""
import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data as data
from loguru import logger


import utils.config as config
from engine.engine import test, etest
from model import build_segmenter
from utils.dataloader import TestDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(
        description='TBNet')
    parser.add_argument('--config',
                        default='config/TBNet.yaml',
                        type=str,
                        help='config file',)
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


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.map_save_path, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # build model
    model, _ = build_segmenter(args)
    model = model.cuda()
    
    args.model_dir = os.path.join(args.output_dir, args.model + '.pth')
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir, map_location="cuda:0")
        if args.model == 'ACUMEN':
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    log_file = "eval_results.txt"

    # load data
    for cur_dataset in args.test_dataset:
        test_root = os.path.join(args.test_root, cur_dataset)
        print(f"Loading {cur_dataset}...")
        test_data = TestDataset(image_root=test_root + '/Imgs/',
                                gt_root=test_root + '/GT/',
                                testsize=args.input_size)
        
        test_loader = data.DataLoader(test_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    drop_last=False)
        if args.visualize:
            args.vis_dir = os.path.join(args.output_dir, "vis", cur_dataset)
            os.makedirs(args.vis_dir, exist_ok=True)

        print(f"Loading {cur_dataset} done.")

        # inference
        results = etest(test_loader, model, cur_dataset, args)
        print(results)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Dataset: {cur_dataset}\n")
            f.write(str(results))
            f.write("\n")
            f.write("-" * 40 + "\n")

if __name__ == '__main__':
    main()
