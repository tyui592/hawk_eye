"""Configuration Code."""
import argparse
from pathlib import Path

import torch
import random
import numpy

from utils.misc import save_pickle, load_pickle


def build_parser():
    """Build arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'val', 'demo', 'video_demo'],
                        default='train')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')
    parser.add_argument('--task',
                        type=str,
                        choices=['ball', 'court', 'bounce'],
                        default='court')
    parser.add_argument('--random-seed',
                        type=int,
                        default=777)

    # model
    parser.add_argument('--model',
                        type=str,
                        choices=['hourglass'],
                        default='hourglass')
    
    parser.add_argument('--num_seq',
                        type=int,
                        help='number of frames for a sequence',
                        default=1)
    
    parser.add_argument('--stride',
                        type=int,
                        default=4,
                        help='downsampling scale from image to heatmap')
    
    parser.add_argument('--num_class',
                        type=int,
                        default=14,
                        help='number of target keypoints, court:14, ball:1')
    
    parser.add_argument('--num_stack',
                        type=int,
                        default=1)
    
    parser.add_argument('--hourglass_inch',
                        type=int,
                        default=128)
    
    parser.add_argument('--increase_ch',
                        type=int,
                        default=0)
    
    parser.add_argument('--activation',
                        type=str,
                        choices=['ReLU', 'LReLU', 'PReLU', 'Linear', 'Sigmoid'],
                        default='ReLU')
    
    parser.add_argument('--pool',
                        type=str,
                        choices=['Max', 'Avg', 'Conv', 'SPP', 'None'],
                        default='Max')
    
    parser.add_argument('--neck_activation',
                        type=str,
                        choices=['ReLU', 'LReLU', 'PReLU', 'Linear', 'Sigmoid'],
                        default='ReLU')
    
    parser.add_argument('--neck_pool',
                        type=str,
                        choices=['Max', 'Avg', 'Conv', 'SPP', 'None'],
                        default='None')
    # postprocessing
    parser.add_argument('--refine_flag',
                        type=int,
                        choices=[0, 1, 2, 3],
                        help='Court edge refine flag, 0: no refine, 1: refine with line, 2: refine with line and homography, 3: refine with homography',
                        default=0)
    
    # data
    parser.add_argument('--data_path',
                        type=Path,
                        required=True,
                        help='Data root')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--imsize',
                        nargs='+',
                        type=int,
                        default=[512, 910],
                        help='image size(height, width)')
    parser.add_argument('--aug_policy',
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='data augmentation policy')
    parser.add_argument('--gaussian_radius',
                        type=float,
                        default=5.0,
                        help='Radius for keypoint gaussian')
    
    # loss
    parser.add_argument('--focal_alpha',
                        type=float,
                        default=2.0,
                        help='Radius for keypoint gaussian')
    parser.add_argument('--focal_beta',
                        type=float,
                        default=4.0,
                        help='Radius for keypoint gaussian')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001)
    
    # train
    parser.add_argument('--epoch',
                        type=int,
                        default=50)

    parser.add_argument('--lr',
                        type=float,
                        default=5e-4)

    parser.add_argument('--lr_gamma',
                        type=float,
                        default=0.1)

    parser.add_argument('--lr_milestone',
                        nargs='+',
                        type=int,
                        default=[40])

    parser.add_argument('--image_set',
                        type=str,
                        default='train',
                        choices=['train', 'val'],
                        help='Image for model training or validation')

    parser.add_argument('--save_root',
                        type=Path,
                        default=None,
                        help='Save root')
    
    # for validation and demo
    parser.add_argument('--model_path',
                        type=Path,
                        default=None,
                        help='Model weight path')

    return parser.parse_args()

# For model evaluation.
MODEL_SPEC = [
    'model', 'num_seq', 'num_class', 'num_stack', 'hourglass_inch', 'increase_ch',
    'activation', 'pool', 'neck_activation', 'neck_pool', 'task', 'imsize',
]

def get_arguments():
    """Get arugments."""
    args = build_parser()

    if args.mode == 'train':
        if args.random_seed is not None:
            # set random seed for reproducible experiments
            # reference: https://github.com/pytorch/pytorch/issues/7068
            random.seed(args.random_seed)
            numpy.random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)

            # these flags can affect performance, select carefully
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        # save arguments
        args.save_root.mkdir(exist_ok=True, parents=True)
        weight_path = args.save_root / 'weights'
        weight_path.mkdir(exist_ok=True)
        save_pickle(args.save_root / 'arguments.pkl', args)
    else:
        # load args from the target experiment dir
        # and update args for load model weights.
        dir_path = args.model_path.parent.parent
        arg_path = dir_path / 'arguments.pkl'
        loaded_args = load_pickle(arg_path)
        args = copy_target_args(loaded_args, args, MODEL_SPEC)
        args.image_set = 'val'

        if args.save_root is None:
            args.save_root = dir_path / f'{args.mode}'
        args.save_root.mkdir(exist_ok=True, parents=True)
    
    return args

def copy_target_args(src, dst, targets):
    """Copy target arguments.

    dst.arg1 = src.arg1
    """
    # namespace to dict
    _src = vars(src)
    _dst = vars(dst)
    for target in targets:
        if target in _src:
            _dst[target] = _src[target]
    return argparse.Namespace(**_dst)