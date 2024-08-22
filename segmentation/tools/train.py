# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
import torch
# from segmentation.models import dino_vision_transformer
# from segmentation.heads import vit_linear_head
import segmentation.datasets
import segmentation.models


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    return args


def update_cfg(cfg, extra_args, args):
    cfg['model']['pretrained'] = extra_args['backbone_weights']
    cfg['model']['backbone']['frozen_stages'] = extra_args['mmseg_frozen_stages']
    cfg['train_dataloader']['batch_size'] = int(extra_args['mmseg_samples_per_gpu'])
    cfg['train_dataloader']['num_workers'] = int(extra_args['mmseg_workers_per_gpu'])
    cfg['optimizer']['lr'] = extra_args['mmseg_lr']
    cfg['optim_wrapper']['optimizer']['lr'] = extra_args['mmseg_lr']
    cfg['optim_wrapper']['optimizer']['weight_decay'] = extra_args['mmseg_wd']
    cfg['param_scheduler'][0]['eta_min'] = float(extra_args['mmseg_lr']) / 10.

    # Multiple tokens
    if cfg['model']['decode_head']['type'] == 'FCNHead' and extra_args['mmseg_ntokens']:
        last_index = cfg['model']['backbone']['out_indices'][0]
        cfg['model']['backbone']['out_indices'] = [last_index - i for i in range(extra_args['mmseg_ntokens'])][::-1]
        cfg['model']['decode_head']['in_index'] = list(range(extra_args['mmseg_ntokens']))
        embed_dim = cfg['model']['decode_head']['in_channels'][0]
        in_channels = extra_args['mmseg_ntokens'] * [embed_dim]
        cfg['model']['decode_head']['in_channels'] = in_channels
        cfg['model']['decode_head']['channels'] = sum(in_channels)

    return cfg


def main(args, extra_args=None):
    # load config
    print(args.config)
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    print('Local rank:', args.local_rank)
    print('Launcher:', args.launcher)

    # Update config
    if extra_args:
        cfg = update_cfg(cfg, extra_args, args)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
