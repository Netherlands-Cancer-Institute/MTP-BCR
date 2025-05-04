import os
import argparse
import json
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
import sys
from utils.mylogging import open_log
from .dataset.data_loader import risk_dataloador


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--arch', default='resnet18',
                        help='only support resnet18 now')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--test_results_dir', type=str, metavar='PATH',
                        default='',
                        help='path to cache (default: none)')
    parser.add_argument('--path_risk_model', type=str, metavar='PATH',
                        default='./weights/best_10y_with_risk_factor',
                        help='path to cache (default: none)')
    parser.add_argument('--csv_path',
                        default='./custom_dataset.csv',
                        type=str, metavar='PATH', help='path to csv (default: none)')
    parser.add_argument('--image_dir', type=str, metavar='PATH',
                        default='',
                        help='path to image data (default: none)')
    parser.add_argument('--num-classes', default='6', type=int, metavar='N',
                        help='(number of classes) predicting BC risk of when training')
    parser.add_argument('--years_of_history', default='21', type=int, metavar='N',
                        help='predicting history of previous tumor')
    parser.add_argument('--test-num-classes', default='11', type=int, metavar='N',
                        help='(number of classes) predicting BC risk of when test')
    parser.add_argument('--num-workers', default='16', type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--img_size', type=int, nargs='+', default=[1024, 512],
                        help='height and width of image in pixels. [default: [256,256]')
    parser.add_argument('--method',
                        default='side_specific_4views_mtp_tumor',
                        help='4views_mtp_tumor, side_specific_4views_mtp_tumor')
    parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
                        default=True,
                        help='use pre-trained ImageNet model')
    # MTP model params
    # ---------------------------------
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--feedforward_dim", type=int, default=256)
    parser.add_argument("--drop_transformer", type=float, default=0.5)
    parser.add_argument("--drop_cpe", type=float, default=0.5)
    parser.add_argument("--pooling", choices=["last_timestep", "sum"],
                        default="last_timestep")
    parser.add_argument("--image_shape", default=(1, 1),
                        help='image shape before input to transformer, '
                             'we set (1,1) as it is already pooled after the encoder for reducing computational consumption'
                        )
    parser.add_argument('--num_time_points', default=6, type=int, metavar='N',
                        help=' num of time points of exam will be input')
    parser.add_argument('--history_selct_method', default='random', type=str,
                        help='last, random method for selecting different time point from history exam')
    # ---------------------------------
    # use risk factors
    # ---------------------------------
    parser.add_argument('--use_risk_factors', action='store_true',
                        # default=True,
                        help='weather balance label')

    parser.add_argument('--multi_tasks', action='store_true',
                        # default=True,
                        help='weather balance label')
    # ---------------------------------
    # For data cleanign
    # ---------------------------------
    parser.add_argument('--missing_clean', action='store_true', default=True,
                        help='clean exams with missing image')

    parser.add_argument('--birads_clean', action='store_true',
                        # default=True,
                        help='clean exams with abnormal finding (bi-Rads > 2)')

    parser.add_argument('--biopsy_clean', action='store_true',
                        # default=True,
                        help='clean exams with abnormal finding (bi-Rads > 2)')

    parser.add_argument('--years_at_least_followup', default=0, type=int, metavar='N',
                        help=' clean exams without enough followup')

    parser.add_argument('--clean_data_without_enough_followup', action='store_true', default=True,
                        help=' clean data without enough followup when computing AUC')
    # ---------------------------------
    # For 10 years risk of BCE method
    # ---------------------------------
    parser.add_argument('--BCE_method', action='store_true', default=True,
                        help='clean exams with abnormal finding (Bi-Rads > 2)')
    # ---------------------------------
    # For 10 years risk of BCE method
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    os.makedirs(args.test_results_dir, exist_ok=True)

    open_log(args.test_results_dir, name='mtpbcr_custom')

    with open(args.path_risk_model + '/args.json', 'r') as f:
        raw_dict = json.load(f)

    args.num_classes = raw_dict['num_classes']
    args.test_num_classes = raw_dict['test_num_classes']
    args.years_of_history = raw_dict['years_of_history']
    args.method = raw_dict['method']
    args.pooling = raw_dict['pooling']
    args.num_time_points = raw_dict['num_time_points']
    args.use_risk_factors = raw_dict['use_risk_factors']
    args.multi_tasks = raw_dict['multi_tasks']
    args.years_at_least_followup = raw_dict['years_at_least_followup']

    if args.BCE_method:
        if args.method == 'side_specific_4views_mtp_tumor':
            from learning_demo.predicting_demo import get_predicting_demo
            from dataset.custom_dataset import Custom_risk_stp_Dataset as Custom_Dataset
            from models.mtpbcr_model import MTP_BCR_Model as risk_model
        else:
            raise ValueError(f" Method: {args.method} is not supported.")

    test = get_predicting_demo(args.method)
    logging.info('finish data loader')
    model = risk_model(args).cuda()
    cudnn.benchmark = True

    checkpoint = torch.load(args.path_risk_model + '/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    os.makedirs(args.test_results_dir, exist_ok=True)

    logging.info('------------ Predict ------------')
    test_data_info = pd.read_csv(args.csv_path)
    test_loader = risk_dataloador(Custom_Dataset, test_data_info, args)

    test(model, test_loader, args, poltroc=True, name='predict')


if __name__ == '__main__':
    main()
