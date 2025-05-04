import math
import gc
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import logging
from sklearn import metrics
from tqdm import tqdm
from ..utils.utils_robust_custom import polt_roc_curve_with_CI

# ---------------------------------
def soft_max_prob(all_prob_risk, all_pred_risk, pred_risk):
    pred_risk = F.softmax(pred_risk, dim=1)
    all_prob_risk.append(pred_risk.cpu().numpy())
    _, risk_predicted = torch.max(pred_risk.data, 1)
    all_pred_risk.append(risk_predicted.cpu().numpy())
    # risk_total_top1 += (risk_predicted == label_risk).float().sum().item()
    return all_prob_risk, all_pred_risk, pred_risk


def eval_auc(
        args,
        all_label_risk,
        all_prob_risk,
        all_followups,
        poltroc,
        name,    # name of roc curve image
        level='breast'  # exam or breast level
             ):
    risk_AUC, c_index, decile_recall = polt_roc_curve_with_CI(
        args, all_label_risk, all_prob_risk, all_followups, poltroc, "{}_{}".format(name, level))

    for cl in range(args.test_num_classes - 1):
        # print('risk within {:.0f} years AUC is'.format(args.test_num_classes - cl - 1), risk_AUC[cl])
        logging.info('***********************   {}-level risk predict ***********************'.format(level))
        logging.info('risk within {:.0f} years '
                     'AUC is {:.4f} '
                     'Bootstrap mean AUC is {:.4f} [{:.4f}, {:.4f}]'.format(
            (args.test_num_classes - cl - 1), risk_AUC[cl][0], risk_AUC[cl][1], risk_AUC[cl][2], risk_AUC[cl][3]))
    logging.info('mean risk is AUC {}'.format(np.mean(risk_AUC, axis=0)[1]))
    logging.info('c-index is {:.4f} Bootstrap mean c-index is {:.4f} [{:.4f}, {:.4f}]'.format(
        c_index[0], c_index[1], c_index[2], c_index[3]))
    logging.info('decile_recall is {:.4f} Bootstrap mean decile_recall is {:.4f} [{:.4f}, {:.4f}]'.format(
        decile_recall[0], decile_recall[1], decile_recall[2], decile_recall[3]))

    for cl in range(args.test_num_classes - 1):
        # print('risk within {:.0f} years AUC is'.format(args.test_num_classes - cl - 1), risk_AUC[cl])
        logging.info('risk within {:.0f} years '
                     'AUC is {:.2f} '
                     'Bootstrap mean AUC is {:.2f} [{:.2f}, {:.2f}]'.format(
            (args.test_num_classes - cl - 1), risk_AUC[cl][0], risk_AUC[cl][1], risk_AUC[cl][2], risk_AUC[cl][3]))
    logging.info('c-index is {:.2f} Bootstrap mean c-index is {:.2f} [{:.2f}, {:.2f}]'.format(
        c_index[0], c_index[1], c_index[2], c_index[3]))
    logging.info('decile_recall is {:.2f} Bootstrap mean decile_recall is {:.2f} [{:.2f}, {:.2f}]'.format(
        decile_recall[0], decile_recall[1], decile_recall[2], decile_recall[3]))
    logging.info('*****************************************************************************'.format(level))

    return risk_AUC, c_index, decile_recall


def predicting_(model, test_loader, args, poltroc=False, name='best model'):
    model.eval()
    risk_total_top1, total_num_patient = 0.0, 0
    all_pred_risk, all_pred_risk_r, all_pred_risk_l, all_pred_risk_breast, = [], [], [], []
    all_prob_risk, all_prob_risk_r, all_prob_risk_l,  all_prob_risk_breast, = [], [], [], []
    all_label_risk, all_label_risk_r, all_label_risk_l, all_label_risk_breast, = [], [], [], []

    all_followups, all_followups_breast = [], []

    save_dict = {
        'patient_id': [],
        'exam_id': [],
        'pred_risk': [],
        'pred_risk_r': [],
        'pred_risk_l': [],
        'label_risk': [],
        'label_risk_r': [],
        'label_risk_l': [],
        'years_last_followup': [],
        'risk_AUC_exam': [],
        'c_index_exam': [],
        'decile_recall_exam': [],
        'risk_AUC_breast': [],
        'c_index_breast': [],
        'decile_recall_breast': [],
    }

    with torch.no_grad():
        valid_bar = tqdm(test_loader)
        # for data, label, _ in valid_bar:
        debug_i = 0
        for input in valid_bar:
            # debug_i += 1
            # if debug_i >= 300:
            #     break
            save_dict['patient_id'].append(input['patient_id'])
            save_dict['exam_id'].append(input['exam_id'])
            save_dict['label_risk'].append(input['risk'])
            save_dict['label_risk_r'].append(input['right_risk'])
            save_dict['label_risk_l'].append(input['left_risk'])
            save_dict['years_last_followup'].append(input['follow_up'])
            # input
            imgs = input['imgs'].cuda()
            times = input['times'].cuda()
            lens = input['lens'].cuda()
            labels = input['labels']
            label_risk = input['risk'].cuda()
            label_risk_r = input['right_risk'].cuda()
            label_risk_l = input['left_risk'].cuda()
            years_last_followup = input['follow_up'].cuda()
            # tumor_infos = input['tumor_infos']
            m_tumor_infos = input['m_tumor_infos'].cuda()

            all_label_risk.append(label_risk.cpu().numpy())
            all_label_risk_r.append(label_risk_r.cpu().numpy())
            all_label_risk_l.append(label_risk_l.cpu().numpy())
            all_label_risk_breast.append(label_risk_r.cpu().numpy())
            all_label_risk_breast.append(label_risk_l.cpu().numpy())

            all_followups.append(years_last_followup.cpu().numpy())
            all_followups_breast.append(years_last_followup.cpu().numpy())
            all_followups_breast.append(years_last_followup.cpu().numpy())

            if args.use_risk_factors:
                input_risk_factors = torch.cat([
                    torch.unsqueeze(labels['race'], 1),
                    torch.unsqueeze(labels['gene'], 1),
                    torch.unsqueeze(labels['menarche'], 1),
                    torch.unsqueeze(labels['menopausal'], 1),
                    torch.unsqueeze(labels['ovarian_cancer'], 1),
                    torch.unsqueeze(labels['family_history'], 1),
                    torch.unsqueeze(labels['family_history_breast'], 1),
                    torch.unsqueeze(labels['family_history_other'], 1),
                    torch.unsqueeze(labels['family_history_ovarian'], 1),
                    # torch.squeeze(tumor_infos, 1),
                ], 1)
                # print('input_risk_factors sizee', input_risk_factors.size())
                # input_risk_factors = input_risk_factors.cuda()
                # input_risk_factors = torch.unsqueeze(input_risk_factors, 0)
                input_risk_factors = input_risk_factors.type(torch.FloatTensor).cuda()
                # print('input_risk_factors sizee', input_risk_factors.size())
                # pred = model(imgs, times, lens, input_risk_factors)
                pred = model(imgs, times, lens, input_risk_factors, m_tumor_infos)
            else:
                pred = model(imgs, times, lens)

            pred_risk = pred['pred_risk']
            pred_risk_r = pred['pred_risk_r']
            pred_risk_l = pred['pred_risk_l']

            total_num_patient += imgs.size(0)

            all_prob_risk_r, all_pred_risk_r, pred_risk_r_ = soft_max_prob(all_prob_risk_r, all_pred_risk_r, pred_risk_r)
            all_prob_risk_l, all_pred_risk_l, pred_risk_l_ = soft_max_prob(all_prob_risk_l, all_pred_risk_l, pred_risk_l)

            all_prob_risk, all_pred_risk, pred_risk_ = soft_max_prob(all_prob_risk, all_pred_risk, pred_risk)

            save_dict['pred_risk'].append(pred_risk_)
            save_dict['pred_risk_r'].append(pred_risk_r_)
            save_dict['pred_risk_l'].append(pred_risk_l_)

            all_prob_risk_breast, all_pred_risk_breast, _ = soft_max_prob(
                all_prob_risk_breast, all_pred_risk_breast, pred_risk_r)

            all_prob_risk_breast, all_pred_risk_breast, _ = soft_max_prob(
                all_prob_risk_breast, all_pred_risk_breast, pred_risk_l)

        risk_AUC_exam, c_index_exam, decile_recall_exam = eval_auc(
            args, all_label_risk, all_prob_risk, all_followups, poltroc, name, level='exam')

        save_dict['risk_AUC_exam'] = risk_AUC_exam
        save_dict['c_index_exam'] = c_index_exam
        save_dict['decile_recall_exam'] = decile_recall_exam

        risk_AUC_breast, c_index_breast, decile_recall_breast = eval_auc(
            args, all_label_risk_breast, all_prob_risk_breast, all_followups_breast, poltroc, name, level='breast')

        save_dict['risk_AUC_breast'] = risk_AUC_breast
        save_dict['c_index_breast'] = c_index_breast
        save_dict['decile_recall_breast'] = decile_recall_breast

        pickle.dump(save_dict, open('{}/result_{}.pkl'.format(args.test_results_dir, name), 'wb'))


def get_predicting_demo(model_method):
    if model_method == '4views_mtp_tumor' or model_method == 'side_specific_4views_mtp_tumor':
        return predicting_
    else:
        raise AttributeError('model method: {}'.format(model_method))
