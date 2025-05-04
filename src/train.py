import os
import argparse
import json
import torch
import logging
import numpy as np
from utils.mylogging import open_log
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from dataset.data_loader import risk_dataloador
from datetime import datetime
from utils.utils_robust_custom import save_checkpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--arch', default='resnet18',
                        help='only support resnet18 as the backbone now')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='optimizer')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--schedule', default=[10, 20, 30, 40], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W',
                        help='weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./log/Mammo_risk/', type=str, metavar='PATH',
                        help='path to cache (default: none)')
    parser.add_argument('--csv-dir', default='', type=str, metavar='PATH',
                        help='path to csv (default: none)')
    parser.add_argument('--image-dir', default='', type=str, metavar='PATH',
                        help='path to image data (default: none)')
    parser.add_argument('--num-classes', default='6', type=int, metavar='N',
                        help='(number of classes) predicting BC risk of when training')
    parser.add_argument('--years_of_history', default='21', type=int, metavar='N',
                        help='predicting history of previous tumor')
    parser.add_argument('--test-num-classes', default='6', type=int, metavar='N',
                        help='(number of classes) predicting BC risk of when test')
    parser.add_argument('--num-workers', default='16', type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--img-size', type=int, nargs='+', default=[1024, 512],
                        help='height and width of image in pixels. [default: [256,256]')
    parser.add_argument('--method',
                        default='side_specific_4views_mtp_tumor',
                        help='4views_mtp_tumor, side_specific_4views_mtp_tumor')
    parser.add_argument('--dataset', default='inhouse',
                        help='inhouse or custom dataset')
    parser.add_argument('--name', default='', type=str,
                        help='neme')
    parser.add_argument('--from-imagenet', default=True, dest='from_imagenet', action='store_true',
                        help='use pre-trained ImageNet model')

    # cross val
    # ---------------------------------
    parser.add_argument('--folds', default=1, type=int, metavar='N',
                        help='x-fold cross-validation, suppose 1 5 10')
    # ---------------------------------

    # Weights of different label when compute loss
    # ---------------------------------
    parser.add_argument('--lambda-risk', default=1.0, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda-age', default=0.2, type=float, metavar='M',
                        help='* 0.0002, 0.2')
    parser.add_argument('--lambda-birads', default=0.2, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda-history', default=0.2, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda-view', default=0.2, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda_bound_union', default=0.000, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda_supcon', default=0.0, type=float, metavar='M',
                        help='')
    parser.add_argument('--lambda_sv', default=0.00, type=float, metavar='M',
                        help='')
    # ---------------------------------

    # Reuse the same splitted csv
    # ---------------------------------
    parser.add_argument('--reuse_csv', type=str, metavar='PATH', default='',
                        help='path to latest checkpoint (default: none)')
    # ---------------------------------

    # Reuse the same balance csv
    # ---------------------------------
    parser.add_argument('--balance', action='store_true',
                        help='weather balance label')
    parser.add_argument('--label_bal', default='pos', type=str,
                        help='[pos]/negative (get BC within 5 Y) label or [risk]')
    # ---------------------------------

    # MTP model params
    # ---------------------------------
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--feedforward_dim", type=int, default=256)
    parser.add_argument("--drop_transformer", type=float, default=0.5)
    parser.add_argument("--drop_cpe", type=float, default=0.5)
    parser.add_argument("--pooling", choices=["last_timestep", "sum"],
                        default="last_timestep",
                        # default="sum",
    )
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
                        help='weather use risk factors')

    parser.add_argument('--multi_tasks', action='store_true',
                        # default=True,
                        help='weather balance label')
    # ---------------------------------

    # For data cleaning
    # ---------------------------------
    parser.add_argument('--missing_clean', action='store_true',
                        default=True,
                        help='clean exams with missing image')

    parser.add_argument('--birads_clean', action='store_true',
                        # default=True,
                        help='clean exams with abnormal finding (bi-Rads > 2)')

    parser.add_argument('--biopsy_clean', action='store_true',
                        # default=True,
                        help='clean exams with abnormal finding (bi-Rads > 2)')

    parser.add_argument('--years_at_least_followup', default=5, type=int, metavar='N',
                        help=' clean exams without enough followup')

    parser.add_argument('--clean_data_without_enough_followup', action='store_true',
                        default=True,
                        help=' clean data without enough followup when computing AUC')
    # ---------------------------------

    args = parser.parse_args()

    if args.balance:
        if args.lambda_age == 0 and args.lambda_birads == 0 and args.lambda_history == 0:
            if '4views_mtp' in args.method:
                args.name = 'baseline_{}_{}_balanced_{}_history_selct_method_{}pooling'.format(
                    args.method, args.label_bal, args.history_selct_method, args.pooling)
            else:
                args.name = 'baseline_{}_{}_balanced'.format(args.method, args.label_bal)
        else:
            if '4views_mtp' in args.method:
                args.name = 'multitask_{}_{}_balanced_{}_history_selct_method_{}pooling'.format(
                    args.method, args.label_bal, args.history_selct_method, args.pooling)
            else:
                args.name = 'multitask_{}_{}_balanced'.format(args.method, args.label_bal)
    else:
        if args.lambda_age == 0 and args.lambda_birads == 0 and args.lambda_history == 0:
            if '4views_mtp' in args.method:
                args.name = 'baseline_{}_imbalanced_{}_history_selct_method_{}pooling'.format(
                    args.method, args.history_selct_method, args.pooling)
            else:
                args.name = 'baseline_{}_imbalanced'.format(args.method)
        else:
            if '4views_mtp' in args.method:
                args.name = 'multitask_{}_imbalanced_{}_history_selct_method_{}pooling'.format(
                    args.method, args.history_selct_method, args.pooling)
            else:
                args.name = 'multitask_{}_imbalanced'.format(args.method)

    args.results_dir = '{}{}-fold-cross-validation/{}_year_risk/{}/{}_{}epochs_batch_size_{}_{}/'.format(
        args.results_dir, args.folds, args.test_num_classes-1, args.name, args.dataset,
        args.epochs, args.batch_size, datetime.now().strftime("%Y-%m-%d-%H-%M"))

    os.makedirs(args.results_dir, exist_ok=True)
    print(str(args).replace(',', "\n"))
    return args


def main():
    args = arg_parse()

    if args.method == '4views_mtp_tumor' or args.method == 'side_specific_4views_mtp_tumor':
        from learning_demo.mtpbcr_task_specific_BCE_demo import get_train_val_test_demo
        from models.mtpbcr_model import MTP_BCR_Model as risk_model
    else:
        raise ValueError(f" Method: {args.method} is not supported.")

    if args.dataset == 'inhouse':
        from dataset.inhouse_MTP_dataset import inhouse_Dataset
    elif args.dataset == 'custom':
        raise ValueError(" Custom dataset is not supported for training yet.")
    else:
        raise ValueError(f" dataset: {args.dataset} is not supported.")

    train, validate, test = get_train_val_test_demo(args.method)

    test_data_info = pd.read_csv(args.reuse_csv + '/test_data_info.csv')
    train_data_info_ = []
    valid_data_info_ = []
    for fold in range(args.folds):
        train_data_info_.append(pd.read_csv(args.reuse_csv + '/train_data_info_{}_fold.csv'.format(fold)))
        valid_data_info_.append(pd.read_csv(args.reuse_csv + '/valid_data_info_{}_fold.csv'.format(fold)))

    test_data_info.to_csv(args.results_dir + '/test_data_info.csv')
    test_loader = risk_dataloador(
        inhouse_Dataset,
        test_data_info,
        args,
        train=False,
        train_transform_method=False,
        RandomSampler_method=False
    )

    for fold in range(args.folds):
        train_data_info = train_data_info_[fold]
        valid_data_info = valid_data_info_[fold]
        train_data_info.to_csv(args.results_dir + '/train_data_info_{}_fold.csv'.format(fold))
        valid_data_info.to_csv(args.results_dir + '/valid_data_info_{}_fold.csv'.format(fold))

    for fold in range(args.folds):
        # best_acc = 0
        best_auc = 0

        args.results_dir_fold = args.results_dir + 'fold_{}/'.format(fold)
        os.makedirs(args.results_dir_fold, exist_ok=True)

        open_log(args)
        logging.info(str(args).replace(',', "\n"))

        train_loader = risk_dataloador(
            inhouse_Dataset,
            train_data_info_[fold],
            args,
            train=True,
            train_transform_method=True,
            RandomSampler_method=args.balance
        )

        valid_loader = risk_dataloador(
            inhouse_Dataset,
            valid_data_info_[fold],
            args,
            train=False,
            train_transform_method=False,
            RandomSampler_method=False
        )

        logging.info('finish data loader')
        model = risk_model(args).cuda()
        logging.info(model)

        # define loss function (criterion) and optimizer
        criterion = {
            'ce_criterion': nn.CrossEntropyLoss(ignore_index=-1),
            'l1_criterion': nn.L1Loss()
        }

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        epoch_start = args.start_epoch

        # if args.resume is not '':
        #     checkpoint = torch.load(args.resume)
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     epoch_start = checkpoint['epoch'] + 1
        #     best_auc = checkpoint['best_auc']
        #     print('Loaded from: {}'.format(args.resume))

        cudnn.benchmark = True

        log_results = {'epoch': [], 'train_loss': [], 'train_acc': [],
                       'valid_loss': [], 'valid_acc': [], 'CM': [], }

        results = {'Name': [], 'ACC': [], 'CM': [], }

        for cl in range(args.num_classes - 1):
            dict_add = {'valid_{:.0f}_year_auc'.format(args.num_classes - cl - 1): []}
            log_results.update(dict_add)

        for cl in range(args.test_num_classes - 1):
            dict_add = {'{:.0f}_year_auc'.format(args.test_num_classes - cl - 1): []}
            results.update(dict_add)

        # dump args
        with open(args.results_dir_fold + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

        # training loop
        for epoch in range(epoch_start, args.epochs + 1):
            if epoch == 0:
                test_loss, test_acc, test_auc, CM = test(
                    model,
                    test_loader,
                    criterion,
                    args,
                    poltroc=True,
                    name='Init Test'
                )
                results['Name'].append('Init Test')
                results['ACC'].append(test_acc)
                for cl in range(args.test_num_classes - 1):
                    results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
                    logging.info('Init Model in test dataset, risk within {:.0f} years AUC is {:.4f} '
                                 'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'.format(
                        (args.test_num_classes - cl - 1),
                        test_auc[cl][0],
                        test_auc[cl][1],
                        test_auc[cl][2],
                        test_auc[cl][3])
                    )

                results['CM'].append(CM)
                logging.info('Init Model in test dataset, Test ACC :{} Test mAUC is :{}'.format(
                    test_acc, np.mean(test_auc, axis=0)[1]))
                # print('Init Model in test dataset, Test ACC :', test_acc, 'Test AUC :', test_auc,
                #       'Test mAUC is :', sum(test_auc) / len(test_auc))
            else:
                log_results['epoch'].append(epoch)
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, args)
                log_results['train_loss'].append(train_loss)
                log_results['train_acc'].append(train_acc)
                valid_loss, valid_acc, valid_auc, CM = \
                    validate(model, valid_loader, criterion, args, poltroc=True, name=epoch)
                log_results['valid_loss'].append(valid_loss)
                log_results['valid_acc'].append(valid_acc)
                for cl in range(args.num_classes - 1):
                    log_results['valid_{:.0f}_year_auc'.format(args.num_classes - cl - 1)].append(valid_auc[cl])
                log_results['CM'].append(CM)
                # save statistics
                data_frame = pd.DataFrame(data=log_results,
                                          # index=range(epoch_start + 1, epoch + 1)
                                          )
                # data_frame = pd.DataFrame(data=log_results)
                data_frame.to_csv(args.results_dir_fold + '/log.csv', index_label='epoch')
                # save model
                # ---------------------------------
                # # remember best acc@1 and save checkpoint
                # is_best = valid_acc > best_acc
                # best_acc = max(valid_acc, best_acc)
                # mean_auc = sum(valid_auc) / len(valid_auc)
                # ---------------------------------

                # ---------------------------------
                # remember best auc and save checkpoint
                mean_auc = sum(valid_auc[args.num_classes - args.test_num_classes:]) / (args.test_num_classes - 1)
                is_best = mean_auc > best_auc
                best_auc = max(mean_auc, best_auc)
                # ---------------------------------

                logging.info('epoch: {} Val ACC is : {} Val mAUC for {} years is  {} Val AUC is :{}'.format(
                    epoch, valid_acc, args.test_num_classes - 1, mean_auc, valid_auc))

                if is_best:
                    logging.info('epoch: {} is test best now, will save in csv, Val ACC is : {} '
                                 'Val mAUC for {} years is  {} Val AUC is :{}'.format(
                        epoch, valid_acc, args.test_num_classes - 1, mean_auc, valid_auc))
                    # print('epoch:', epoch, 'is test best now, will save in csv, Val ACC is :',
                    #       valid_acc, 'Val AUC is :', valid_auc, 'Val mAUC is :', mean_auc)
                    # #################################################################################
                    #
                    # test_loss, test_acc, test_auc, CM = test(model, test_loader, criterion, args, poltroc=True,
                    #                                          name='Best Model Test {} epoch'.format(epoch))
                    # results['Name'].append('Best test {} epoch'.format(epoch))
                    # results['ACC'].append(test_acc)
                    #
                    # for cl in range(args.test_num_classes - 1):
                    #     results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
                    #     logging.info('Model_best in test dataset, risk within {:.0f} years AUC is {:.4f} '
                    #                  'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'.format(
                    #         (args.test_num_classes - cl - 1),
                    #         test_auc[cl][0],
                    #         test_auc[cl][1],
                    #         test_auc[cl][2],
                    #         test_auc[cl][3])
                    #     )
                    #
                    # results['CM'].append(CM)
                    # logging.info('Model_best in test dataset, Test ACC :{} Test mAUC is :{}'.format(
                    #     test_acc, np.mean(test_auc, axis=0)[1]))
                    # #################################################################################

                    # for cl in range(args.test_num_classes - 1):
                    #     results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
                    # results['CM'].append(CM)
                    # # print('Model_best in tset dataset, Test ACC :', test_acc, 'Test AUC :', test_auc,
                    # #       'Test mAUC is :', sum(test_auc) / len(test_auc))
                    # logging.info('Model_best in tset dataset, Test ACC :{} Test AUC : {} Test mAUC is {}:'.format(
                    #     test_acc, test_auc, sum(test_auc) / len(test_auc)))

                if is_best or epoch > (args.epochs - 2):
                    save_checkpoint(
                        args.results_dir_fold,
                        {'epoch': epoch + 1,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_auc': best_auc,
                         'optimizer': optimizer.state_dict(),
                         'log_results': log_results,},
                        is_best)

        #
        checkpoint = torch.load(args.results_dir_fold + '/model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc, test_auc, CM = test(model, test_loader, criterion, args, poltroc=True,
                                                 name='Best Model Test')
        results['Name'].append('Best Model')
        results['ACC'].append(test_acc)

        # for cl in range(args.test_num_classes - 1):
        #     results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
        # results['CM'].append(CM)
        # # print('Model_best in test dataset, Test ACC :', test_acc, 'Test AUC :', test_auc,
        # #       'Test mAUC is :', sum(test_auc) / len(test_auc))
        # logging.info('Model_best in tset dataset, Test ACC :{} Test AUC : {} Test mAUC is {}:'.format(
        #     test_acc, test_auc, sum(test_auc) / len(test_auc)))

        for cl in range(args.test_num_classes - 1):
            results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
            logging.info('Model_best in test dataset, risk within {:.0f} years AUC is {:.4f} '
                         'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'.format(
                (args.test_num_classes - cl - 1),
                test_auc[cl][0],
                test_auc[cl][1],
                test_auc[cl][2],
                test_auc[cl][3])
            )

        results['CM'].append(CM)
        logging.info('Model_best in test dataset, Test ACC :{} Test mAUC is :{}'.format(
            test_acc, np.mean(test_auc, axis=0)[1]))

        checkpoint = torch.load(args.results_dir_fold + '/model_last.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc, test_auc, CM = test(model, test_loader, criterion, args, poltroc=True,
                                                 name='Last Model Test')
        results['Name'].append('Last Model')
        results['ACC'].append(test_acc)

        # for cl in range(args.test_num_classes - 1):
        #     results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
        # results['CM'].append(CM)
        # # print('Model_last in test dataset, Test ACC :', test_acc, 'Test AUC :', test_auc,
        # #       'Test mAUC is :', sum(test_auc) / len(test_auc))
        # logging.info('Model_last in tset dataset, Test ACC :{} Test AUC : {} Test mAUC is {}:'.format(
        #     test_acc, test_auc, sum(test_auc) / len(test_auc)))

        for cl in range(args.test_num_classes - 1):
            results['{:.0f}_year_auc'.format(args.test_num_classes - cl - 1)].append(test_auc[cl])
            logging.info('Model_last in test dataset, risk within {:.0f} years AUC is {:.4f} '
                         'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'.format(
                (args.test_num_classes - cl - 1),
                test_auc[cl][0],
                test_auc[cl][1],
                test_auc[cl][2],
                test_auc[cl][3])
            )

        results['CM'].append(CM)
        logging.info('Model_last in test dataset, Test ACC :{} Test mAUC is :{}'.format(
            test_acc, np.mean(test_auc, axis=0)[1]))

        results_data_frame = pd.DataFrame(data=results)
        results_data_frame.to_csv(args.results_dir_fold + '/results.csv')

if __name__ == '__main__':
    main()
