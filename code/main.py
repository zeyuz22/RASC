from __future__ import print_function

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.backends.cudnn as cudnn
import os.path as osp
import numpy as np
import cv2
import wandb
import importlib
import random
import torch
import rasc

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--budget', default=0.05, type=float, help='budget')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--gpu', default=0, type=str, help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--source', type=str, default='', help="The source dataset path list")
    parser.add_argument('--target', type=str, default='', help="The target dataset path list")
    parser.add_argument('--s', type=str, default='', help="The source dataset name")
    parser.add_argument('--t', type=str, default='', help="The target dataset name")
    parser.add_argument('--target-val', type=str, default='', help="The target validation dataset path list")
    parser.add_argument('--class-num', default=31, type=int, help='class num of dataset')
    parser.add_argument('--n-samples', type=int, default=2, help='number of samples from each class')
    parser.add_argument('--pre-train', default=False, action='store_true')
    parser.add_argument('--cont-t', default=0.1, type=float, help='contrastive loss T')
    parser.add_argument('--cont-loss-weight1', default=0.5, type=float, help='contrastive loss weight 1')
    parser.add_argument('--cont-loss-weight2', default=0.7, type=float, help='contrastive loss weight 2')
    parser.add_argument('--th', type=float, default=0.9)
    parser.add_argument('--m', type=float, default=0.999)
    parser.add_argument('--ema', type=float, default=0.9)
    parser.add_argument('--method', type=str, default='rasc')
    args = parser.parse_args()

    ''' Fix seed '''
    random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.save_model_path = "checkpoint/" + args.s + args.t
    if not osp.exists(args.save_model_path):
        os.system('mkdir -p ' + args.save_model_path)
    if not osp.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    if args.method == 'rasc':
        trainer = rasc.RASC(args)

    if args.pre_train:
        trainer.pre_train()
    else:
        trainer.train()
