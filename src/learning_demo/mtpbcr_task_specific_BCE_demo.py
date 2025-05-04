import math
import gc
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import logging
from sklearn import metrics
from tqdm import tqdm

from ..utils.utils_robust import comput_cm, polt_roc_curve, polt_roc_curve_with_CI, get_y_ture_pred, write_report_to_text


# lr scheduler for training
# ---------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr

    if args.cos:  # cosine lr schedule
        lr *= 0.1 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """
# ---------------------------------

# using BCE loss for images without enough follow up
# ---------------------------------
def get_risk_loss_BCE(pred, risk_label, years_last_followup):
    # criterion = torch.nn.BCELoss().cuda()
    # print('pred.shape', pred.shape)
    pred = F.softmax(pred, dim=1)
    batch_size, num_pred_years = pred.shape
    years_last_followup = years_last_followup.cpu().detach().numpy()
    risk_mask = torch.zeros((batch_size, num_pred_years))
    y_seq = torch.zeros((batch_size, num_pred_years))

    for i in range(batch_size):
        y_seq[i, risk_label[i]] = 1
        if risk_label[i] == 0 and int(years_last_followup[i]) < (num_pred_years-2):
            risk_mask[i, num_pred_years - int(years_last_followup[i]) - 1:] = 1
        else:
            risk_mask[i, :] = 1

    # loss = F.binary_cross_entropy(pred, y_seq.float().cuda(), weight=risk_mask.float().cuda()) * 20
    loss = F.binary_cross_entropy(
        pred, y_seq.float().cuda(),
        weight=risk_mask.float().cuda(),
        reduction='sum'
    ) / torch.sum(risk_mask.float()) * 20
    return loss

# ---------------------------------

# MTP model params add more risk factor (from: )
# ---------------------------------
def label_filter_220609(label, lens, batchsize):
    # print('label_filter_220609')
    # print('lens:', lens)
    # print('label:' ,label)
    #
    label_m = torch.zeros(1)
    for index_ in range(batchsize):
        label_m = torch.cat([label_m, label[index_, :lens[index_]]])

    label = label_m[1:]
    # print('new label:', label)
    return label


def compute_right_side_loss_tumor(args, pred, m_label, lens, ce_criterion, l1_criterion):
    # print('compute_right_side_loss')
    # print('pred', pred)

    risk = pred['pred_risk']
    history = pred['pred_history']
    risks_r_label = label_filter_220609(
        m_label['risks_r'], lens, args.batch_size).long().cuda()
    years_last_followup = label_filter_220609(
        m_label['years_last_followup'], lens, args.batch_size).long().cuda()

    history_label = label_filter_220609(
        m_label['history_r'], lens, args.batch_size).long().cuda()
    # print('risk_loss')
    # risk_loss = ce_criterion(risk, risks_r_label) * 0.5
    risk_loss = get_risk_loss_BCE(risk, risks_r_label, years_last_followup) * 0.5
    # print('history_loss')
    history_loss = ce_criterion(history, history_label) * 0.2
    loss = risk_loss + history_loss

    if args.use_risk_factors:
        location_before = pred['pred_location_before']
        location_next = pred['pred_location_next']
        pred_type_before = pred['pred_type_before']
        pred_type_next = pred['pred_type_next']
        pred_PCR = pred['pred_PCR']
        pred_pT_stage = pred['pred_pT_stage']
        pred_pN_stage = pred['pred_pN_stage']
        pred_pM_stage = pred['pred_pM_stage']
        pred_ER = pred['pred_ER']
        pred_PR = pred['pred_PR']
        pred_Her2 = pred['pred_Her2']
        # risk, history, location_before, location_next, pred_type_before, pred_type_next, \
        # pred_PCR, pred_pT_stage, pred_pN_stage, pred_pM_stage, pred_ER, pred_PR, pred_Her2 \
        #     = torch.split(pred, [args.num_classes, args.years_of_history, 8, 8, 2, 2, 2, 5, 4, 2, 2, 2, 2], 1)
        # print('risks_r_label', m_label['risks_r'])

        previous_cancer_location_label = label_filter_220609(
            m_label['previous_cancer_location_right'], lens, args.batch_size).long().cuda()
        next_cancer_location_label = label_filter_220609(
            m_label['next_cancer_location_right'], lens, args.batch_size).long().cuda()
        previous_cancer_type_label = label_filter_220609(
            m_label['previous_cancer_type_right'], lens, args.batch_size).long().cuda()
        next_cancer_type_label = label_filter_220609(
            m_label['next_cancer_type_right'], lens, args.batch_size).long().cuda()

        previous_cancer_PCR_label = label_filter_220609(
            m_label['previous_cancer_PCR_right'], lens, args.batch_size).long().cuda()
        previous_cancer_pT_stage_label = label_filter_220609(
            m_label['previous_cancer_pT_stage_right'], lens, args.batch_size).long().cuda()
        previous_cancer_pN_stage_label = label_filter_220609(
            m_label['previous_cancer_pN_stage_right'], lens, args.batch_size).long().cuda()
        previous_cancer_pM_stage_label = label_filter_220609(
            m_label['previous_cancer_pM_stage_right'], lens, args.batch_size).long().cuda()
        previous_cancer_ER_label = label_filter_220609(
            m_label['previous_cancer_ER_right'], lens, args.batch_size).long().cuda()
        previous_cancer_PR_label = label_filter_220609(
            m_label['previous_cancer_PR_right'], lens, args.batch_size).long().cuda()
        previous_cancer_Her2_label = label_filter_220609(
            m_label['previous_cancer_Her2_right'], lens, args.batch_size).long().cuda()

        # print('location_next_loss')
        location_next_loss = ce_criterion(location_next, next_cancer_location_label) * 0.2
        # print('location_before_loss')
        location_before_loss = ce_criterion(location_before, previous_cancer_location_label) * 0.2
        # print('type_before_loss')
        type_before_loss = ce_criterion(pred_type_before, previous_cancer_type_label) * 0.2
        # print('type_next_loss')
        type_next_loss = ce_criterion(pred_type_next, next_cancer_type_label) * 0.2

        # print('type_next_loss')
        PCR_loss = ce_criterion(pred_PCR, previous_cancer_PCR_label) * 0.2

        # print('type_next_loss')
        pT_stage_loss = ce_criterion(pred_pT_stage, previous_cancer_pT_stage_label) * 0.2

        # print('type_next_loss')
        pN_stage_loss = ce_criterion(pred_pN_stage, previous_cancer_pN_stage_label) * 0.2

        # print('type_next_loss')
        pM_stage_loss = ce_criterion(pred_pM_stage, previous_cancer_pM_stage_label) * 0.2

        # print('type_next_loss')
        ER_loss = ce_criterion(pred_ER, previous_cancer_ER_label) * 0.2

        # print('type_next_loss')
        PR_loss = ce_criterion(pred_PR, previous_cancer_PR_label) * 0.2

        # print('type_next_loss')
        Her2_loss = ce_criterion(pred_Her2, previous_cancer_Her2_label) * 0.2

        loss = loss + \
               location_next_loss + location_before_loss + \
               type_before_loss + type_next_loss + \
               PCR_loss + pT_stage_loss + pN_stage_loss + pM_stage_loss + \
               ER_loss + PR_loss + Her2_loss

    return loss


def compute_left_side_loss_tumor(args, pred, m_label, lens, ce_criterion, l1_criterion):
    # print('compute_left_side_loss')
    # print('pred', pred)
    risk = pred['pred_risk']
    history = pred['pred_history']
    risks_l_label = label_filter_220609(
        m_label['risks_l'], lens, args.batch_size).long().cuda()
    # print('history_label')
    history_label = label_filter_220609(
        m_label['history_l'], lens, args.batch_size).long().cuda()
    years_last_followup = label_filter_220609(
        m_label['years_last_followup'], lens, args.batch_size).long().cuda()
    # print('risk_loss')
    # risk_loss = ce_criterion(risk, risks_l_label) * 0.5
    risk_loss = get_risk_loss_BCE(risk, risks_l_label, years_last_followup) * 0.5
    # print('history_loss')
    history_loss = ce_criterion(history, history_label) * 0.2

    loss = risk_loss + history_loss

    if args.use_risk_factors:
        location_before = pred['pred_location_before']
        location_next = pred['pred_location_next']
        pred_type_before = pred['pred_type_before']
        pred_type_next = pred['pred_type_next']
        pred_PCR = pred['pred_PCR']
        pred_pT_stage = pred['pred_pT_stage']
        pred_pN_stage = pred['pred_pN_stage']
        pred_pM_stage = pred['pred_pM_stage']
        pred_ER = pred['pred_ER']
        pred_PR = pred['pred_PR']
        pred_Her2 = pred['pred_Her2']
        # risk, history, location_before, location_next, pred_type_before, pred_type_next, \
        # pred_PCR, pred_pT_stage, pred_pN_stage, pred_pM_stage, pred_ER, pred_PR, pred_Her2 \
        #     = torch.split(pred, [args.num_classes, args.years_of_history, 8, 8, 2, 2, 2, 5, 4, 2, 2, 2, 2], 1)

        # print('m_label[risks_l]' , m_label['risks_l'])
        # print('risks_l_label', m_label['risks_l'])

        # print('previous_cancer_location_label')
        previous_cancer_location_label = label_filter_220609(
            m_label['previous_cancer_location_left'], lens, args.batch_size).long().cuda()
        # print('next_cancer_location_label')
        next_cancer_location_label = label_filter_220609(
            m_label['next_cancer_location_left'], lens, args.batch_size).long().cuda()
        # print('previous_cancer_type_label')
        previous_cancer_type_label = label_filter_220609(
            m_label['previous_cancer_type_left'], lens, args.batch_size).long().cuda()
        # print('next_cancer_type_label')
        next_cancer_type_label = label_filter_220609(
            m_label['next_cancer_type_left'], lens, args.batch_size).long().cuda()

        previous_cancer_PCR_label = label_filter_220609(
            m_label['previous_cancer_PCR_left'], lens, args.batch_size).long().cuda()
        previous_cancer_pT_stage_label = label_filter_220609(
            m_label['previous_cancer_pT_stage_left'], lens, args.batch_size).long().cuda()
        previous_cancer_pN_stage_label = label_filter_220609(
            m_label['previous_cancer_pN_stage_left'], lens, args.batch_size).long().cuda()
        previous_cancer_pM_stage_label = label_filter_220609(
            m_label['previous_cancer_pM_stage_left'], lens, args.batch_size).long().cuda()
        previous_cancer_ER_label = label_filter_220609(
            m_label['previous_cancer_ER_left'], lens, args.batch_size).long().cuda()
        previous_cancer_PR_label = label_filter_220609(
            m_label['previous_cancer_PR_left'], lens, args.batch_size).long().cuda()
        previous_cancer_Her2_label = label_filter_220609(
            m_label['previous_cancer_Her2_left'], lens, args.batch_size).long().cuda()

        # print('location_next_loss')
        location_next_loss = ce_criterion(location_next, next_cancer_location_label) * 0.2
        # print('location_before_loss')
        location_before_loss = ce_criterion(location_before, previous_cancer_location_label) * 0.2
        # print('type_before_loss')
        type_before_loss = ce_criterion(pred_type_before, previous_cancer_type_label) * 0.2
        # print('type_next_loss')
        type_next_loss = ce_criterion(pred_type_next, next_cancer_type_label) * 0.2

        # print('type_next_loss')
        PCR_loss = ce_criterion(pred_PCR, previous_cancer_PCR_label) * 0.2

        # print('type_next_loss')
        pT_stage_loss = ce_criterion(pred_pT_stage, previous_cancer_pT_stage_label) * 0.2

        # print('type_next_loss')
        pN_stage_loss = ce_criterion(pred_pN_stage, previous_cancer_pN_stage_label) * 0.2

        # print('type_next_loss')
        pM_stage_loss = ce_criterion(pred_pM_stage, previous_cancer_pM_stage_label) * 0.2

        # print('type_next_loss')
        ER_loss = ce_criterion(pred_ER, previous_cancer_ER_label) * 0.2

        # print('type_next_loss')
        PR_loss = ce_criterion(pred_PR, previous_cancer_PR_label) * 0.2

        # print('type_next_loss')
        Her2_loss = ce_criterion(pred_Her2, previous_cancer_Her2_label) * 0.2

        loss = loss + \
               location_next_loss + location_before_loss + \
               type_before_loss +  type_next_loss + \
               PCR_loss + pT_stage_loss + pN_stage_loss + pM_stage_loss + \
               ER_loss + PR_loss + Her2_loss

    return loss


def compute_exam_based_loss_tumor(args, pred, m_label, lens, ce_criterion, l1_criterion):
    # print('compute_exam_based_loss')
    risk = pred['pred_risk']
    history = pred['pred_history']
    risk_label = label_filter_220609(
        m_label['risks'], lens, args.batch_size).long().cuda()
    history_label = label_filter_220609(
        m_label['history'], lens, args.batch_size).long().cuda()
    years_last_followup = label_filter_220609(
        m_label['years_last_followup'], lens, args.batch_size).long().cuda()
    # print('risk_loss')
    # risk_loss = ce_criterion(risk, risk_label) * 0.5
    risk_loss = get_risk_loss_BCE(risk, risk_label, years_last_followup) * 0.5
    # print('history_loss')
    history_loss = ce_criterion(history, history_label) * 0.2
    loss = risk_loss + history_loss

    if args.multi_tasks:
        manufactor = pred['pred_manufactor']
        age = pred['pred_age']
        density = pred['pred_density']
        birads = pred['pred_birads']

        age_label = label_filter_220609(
            m_label['age'], lens, args.batch_size).cuda()
        age_ig_label = label_filter_220609(
            m_label['age_ig'], lens, args.batch_size).cuda()
        birads_label = label_filter_220609(
            m_label['birads'], lens, args.batch_size).long().cuda()
        density_label = label_filter_220609(
            m_label['density'], lens, args.batch_size).long().cuda()
        manufactor_label = label_filter_220609(
            m_label['ManufacturerModelName'], lens, args.batch_size).long().cuda()

        age = torch.squeeze(age)
        age_ig_label = torch.squeeze(age_ig_label)
        age_label = torch.squeeze(age_label)
        age = age * age_ig_label
        age_label = age_label * age_ig_label

        # print('birads_loss')
        birads_loss = ce_criterion(birads, birads_label) * 0.2
        # print('density_loss')
        density_loss = ce_criterion(density, density_label) * 0.2
        # print('manufactor_loss')
        manufactor_loss = ce_criterion(manufactor, manufactor_label) * 0.2
        # print('age_loss')
        age_loss = l1_criterion(age, age_label) * 0.002

        loss = loss + birads_loss + density_loss + age_loss + manufactor_loss

    if args.use_risk_factors:
        location_before = pred['pred_location_before']
        location_next = pred['pred_location_next']

        position_before = pred['pred_position_before']
        position_next = pred['pred_position_next']

        pred_type_before = pred['pred_type_before']
        pred_type_next = pred['pred_type_next']

        pred_PCR = pred['pred_PCR']
        pred_pT_stage = pred['pred_pT_stage']
        pred_pN_stage = pred['pred_pN_stage']
        pred_pM_stage = pred['pred_pM_stage']
        pred_ER = pred['pred_ER']
        pred_PR = pred['pred_PR']
        pred_Her2 = pred['pred_Her2']

        next_cancer_position_label = label_filter_220609(
            m_label['next_cancer_position'], lens, args.batch_size).long().cuda()
        previous_cancer_position_label = label_filter_220609(
            m_label['previous_cancer_position'], lens, args.batch_size).long().cuda()
        previous_cancer_type_label = label_filter_220609(
            m_label['previous_cancer_type'], lens, args.batch_size).long().cuda()
        next_cancer_type_label = label_filter_220609(
            m_label['next_cancer_type'], lens, args.batch_size).long().cuda()

        previous_cancer_location_label = label_filter_220609(
            m_label['previous_cancer_location'], lens, args.batch_size).long().cuda()
        next_cancer_location_label = label_filter_220609(
            m_label['next_cancer_location'], lens, args.batch_size).long().cuda()

        previous_cancer_PCR_label = label_filter_220609(
            m_label['previous_cancer_PCR'], lens, args.batch_size).long().cuda()
        previous_cancer_pT_stage_label = label_filter_220609(
            m_label['previous_cancer_pT_stage'], lens, args.batch_size).long().cuda()
        previous_cancer_pN_stage_label = label_filter_220609(
            m_label['previous_cancer_pN_stage'], lens, args.batch_size).long().cuda()
        previous_cancer_pM_stage_label = label_filter_220609(
            m_label['previous_cancer_pM_stage'], lens, args.batch_size).long().cuda()
        previous_cancer_ER_label = label_filter_220609(
            m_label['previous_cancer_ER'], lens, args.batch_size).long().cuda()
        previous_cancer_PR_label = label_filter_220609(
            m_label['previous_cancer_PR'], lens, args.batch_size).long().cuda()
        previous_cancer_Her2_label = label_filter_220609(
            m_label['previous_cancer_Her2'], lens, args.batch_size).long().cuda()

        # print('position_next_loss')
        position_next_loss = ce_criterion(position_next, next_cancer_position_label) * 0.2
        # print('position_before_loss')
        position_before_loss = ce_criterion(position_before, previous_cancer_position_label) * 0.2
        # print('type_before_loss')
        type_before_loss = ce_criterion(pred_type_before, previous_cancer_type_label) * 0.2
        # print('type_next_loss')
        type_next_loss = ce_criterion(pred_type_next, next_cancer_type_label) * 0.2

        location_before_loss = ce_criterion(location_before, previous_cancer_location_label) * 0.2
        location_next_loss = ce_criterion(location_next, next_cancer_location_label) * 0.2
        # print('type_next_loss')
        PCR_loss = ce_criterion(pred_PCR, previous_cancer_PCR_label) * 0.2
        # print('type_next_loss')
        pT_stage_loss = ce_criterion(pred_pT_stage, previous_cancer_pT_stage_label) * 0.2
        # print('type_next_loss')
        pN_stage_loss = ce_criterion(pred_pN_stage, previous_cancer_pN_stage_label) * 0.2
        # print('type_next_loss')
        pM_stage_loss = ce_criterion(pred_pM_stage, previous_cancer_pM_stage_label) * 0.2
        # print('type_next_loss')
        ER_loss = ce_criterion(pred_ER, previous_cancer_ER_label) * 0.2
        # print('type_next_loss')
        PR_loss = ce_criterion(pred_PR, previous_cancer_PR_label) * 0.2
        # print('type_next_loss')
        Her2_loss = ce_criterion(pred_Her2, previous_cancer_Her2_label) * 0.2

        loss = loss + position_next_loss + position_before_loss + type_before_loss + \
               type_next_loss + location_before_loss + \
               location_next_loss + PCR_loss + pT_stage_loss + pN_stage_loss + pM_stage_loss + ER_loss + \
               PR_loss + Her2_loss

    return loss


def compute_final_loss_tumor_0614_(args, pred, label, ce_criterion, l1_criterion):
    risk = pred['pred_risk']
    history = pred['pred_history']
    risk_label = label['risks'].cuda()
    history_label = label['history'].cuda()
    years_last_followup = label['years_last_followup'].cuda()
    # print('risk_loss')
    # risk_loss = ce_criterion(risk, risk_label)
    risk_loss = get_risk_loss_BCE(risk, risk_label, years_last_followup)
    # print('history_loss')
    history_loss = ce_criterion(history, history_label) * 0.2
    loss = risk_loss + history_loss

    if args.multi_tasks:
        age = pred['pred_age']
        density = pred['pred_density']
        birads = pred['pred_birads']

        age_label = label['age'].cuda()
        age_ig_label = label['age_ig'].cuda()
        birads_label = label['birads'].cuda()
        density_label = label['density'].cuda()

        age = torch.squeeze(age)
        age_ig_label = torch.squeeze(age_ig_label)
        age_label = torch.squeeze(age_label)
        age = age * age_ig_label
        age_label = age_label * age_ig_label
        # print('birads_loss')
        birads_loss = ce_criterion(birads, birads_label) * 0.2
        # print('density_loss')
        density_loss = ce_criterion(density, density_label) * 0.2
        # print('age_loss')
        age_loss = l1_criterion(age, age_label) * 0.002
        loss = loss + birads_loss + density_loss + age_loss

    if args.use_risk_factors:
        location_next = pred['pred_location_next']
        position_before = pred['pred_position_before']
        position_next = pred['pred_position_next']
        pred_type_before = pred['pred_type_before']
        pred_type_next = pred['pred_type_next']

        next_cancer_position_label = label['next_cancer_position'].cuda()
        previous_cancer_position_label = label['previous_cancer_position'].cuda()
        previous_cancer_type_label = label['previous_cancer_type'].cuda()
        next_cancer_type_label = label['next_cancer_type'].cuda()

        # previous_cancer_location_label = label['previous_cancer_location'].cuda()
        next_cancer_location_label = label['next_cancer_location'].cuda()
        # previous_cancer_PCR_label = label['previous_cancer_PCR'].cuda()
        # previous_cancer_pT_stage_label = label['previous_cancer_pT_stage'].cuda()
        # previous_cancer_pN_stage_label = label['previous_cancer_pN_stage'].cuda()
        # previous_cancer_pM_stage_label = label['previous_cancer_pM_stage'].cuda()
        # previous_cancer_ER_label = label['previous_cancer_ER'].cuda()
        # previous_cancer_PR_label = label['previous_cancer_PR'].cuda()
        # previous_cancer_Her2_label = label['previous_cancer_Her2'].cuda()

        # print('position_next_loss')
        position_next_loss = ce_criterion(position_next, next_cancer_position_label) * 0.5
        # print('position_before_loss')
        position_before_loss = ce_criterion(position_before, previous_cancer_position_label) * 0.2
        # print('type_before_loss')
        type_before_loss = ce_criterion(pred_type_before, previous_cancer_type_label) * 0.2
        # print('type_next_loss')
        type_next_loss = ce_criterion(pred_type_next, next_cancer_type_label) * 0.2

        # location_before_loss = ce_criterion(location_before, previous_cancer_location_label) * 0.2
        location_next_loss = ce_criterion(location_next, next_cancer_location_label) * 0.2
        # print('type_next_loss')
        # PCR_loss = ce_criterion(pred_PCR, previous_cancer_PCR_label) * 0.2
        # print('type_next_loss')
        # pT_stage_loss = ce_criterion(pred_pT_stage, previous_cancer_pT_stage_label) * 0.2
        # print('type_next_loss')
        # pN_stage_loss = ce_criterion(pred_pN_stage, previous_cancer_pN_stage_label) * 0.2
        # print('type_next_loss')
        # pM_stage_loss = ce_criterion(pred_pM_stage, previous_cancer_pM_stage_label) * 0.2

        # print('type_next_loss')
        # ER_loss = ce_criterion(pred_ER, previous_cancer_ER_label) * 0.2

        # print('type_next_loss')
        # PR_loss = ce_criterion(pred_PR, previous_cancer_PR_label) * 0.2

        # print('type_next_loss')
        # Her2_loss = ce_criterion(pred_Her2, previous_cancer_Her2_label) * 0.2

        loss = loss + position_next_loss + position_before_loss + type_before_loss + \
               type_next_loss + location_next_loss

    return loss


def compute_final_loss_tumor_side_specific(args, pred, label, ce_criterion, l1_criterion):
    risk = pred['pred_risk']
    risk_r = pred['pred_risk_r']
    risk_l = pred['pred_risk_l']

    history = pred['pred_history']
    risk_label = label['risks'].cuda()
    risk_r_label = label['risks_r'].cuda()
    risk_l_label = label['risks_l'].cuda()
    history_label = label['history'].cuda()
    years_last_followup = label['years_last_followup'].cuda()
    # print('risk_loss')
    # risk_loss = ce_criterion(risk, risk_label)
    # risk_r_loss = ce_criterion(risk_r, risk_r_label) * 0.1
    # risk_l_loss = ce_criterion(risk_l, risk_l_label) * 0.1

    risk_loss =get_risk_loss_BCE(risk, risk_label, years_last_followup)
    risk_r_loss =get_risk_loss_BCE(risk_r, risk_r_label, years_last_followup) * 0.1
    risk_l_loss = get_risk_loss_BCE(risk_l, risk_l_label, years_last_followup) * 0.1
    # print('history_loss')
    history_loss = ce_criterion(history, history_label) * 0.2
    loss = risk_loss + history_loss + risk_r_loss + risk_l_loss

    if args.multi_tasks:
        age = pred['pred_age']
        density = pred['pred_density']
        birads = pred['pred_birads']

        age_label = label['age'].cuda()
        age_ig_label = label['age_ig'].cuda()
        birads_label = label['birads'].cuda()
        density_label = label['density'].cuda()

        age = torch.squeeze(age)
        age_ig_label = torch.squeeze(age_ig_label)
        age_label = torch.squeeze(age_label)
        age = age * age_ig_label
        age_label = age_label * age_ig_label
        # print('birads_loss')
        birads_loss = ce_criterion(birads, birads_label) * 0.2
        # print('density_loss')
        density_loss = ce_criterion(density, density_label) * 0.2
        # print('age_loss')
        age_loss = l1_criterion(age, age_label) * 0.002
        loss = loss + birads_loss + density_loss + age_loss

    if args.use_risk_factors:
        location_next = pred['pred_location_next']
        location_next_r = pred['pred_location_next_r']
        location_next_l = pred['pred_location_next']

        position_before = pred['pred_position_before']
        position_next = pred['pred_position_next']
        pred_type_before = pred['pred_type_before']

        pred_type_next = pred['pred_type_next']
        pred_type_next_r = pred['pred_type_next_r']
        pred_type_next_l = pred['pred_type_next_l']

        next_cancer_position_label = label['next_cancer_position'].cuda()
        previous_cancer_position_label = label['previous_cancer_position'].cuda()
        previous_cancer_type_label = label['previous_cancer_type'].cuda()

        next_cancer_type_label = label['next_cancer_type'].cuda()
        next_cancer_type_r_label = label['next_cancer_type_right'].cuda()
        next_cancer_type_l_label = label['next_cancer_type_left'].cuda()

        # previous_cancer_location_label = label['previous_cancer_location'].cuda()
        next_cancer_location_label = label['next_cancer_location'].cuda()
        next_cancer_location_r_label = label['next_cancer_location_right'].cuda()
        next_cancer_location_l_label = label['next_cancer_location_left'].cuda()
        # previous_cancer_PCR_label = label['previous_cancer_PCR'].cuda()
        # previous_cancer_pT_stage_label = label['previous_cancer_pT_stage'].cuda()
        # previous_cancer_pN_stage_label = label['previous_cancer_pN_stage'].cuda()
        # previous_cancer_pM_stage_label = label['previous_cancer_pM_stage'].cuda()
        # previous_cancer_ER_label = label['previous_cancer_ER'].cuda()
        # previous_cancer_PR_label = label['previous_cancer_PR'].cuda()
        # previous_cancer_Her2_label = label['previous_cancer_Her2'].cuda()

        # print('position_next_loss')
        position_next_loss = ce_criterion(position_next, next_cancer_position_label) * 0.5
        # print('position_before_loss')
        position_before_loss = ce_criterion(position_before, previous_cancer_position_label) * 0.2
        # print('type_before_loss')
        type_before_loss = ce_criterion(pred_type_before, previous_cancer_type_label) * 0.2
        # print('type_next_loss')
        type_next_loss = ce_criterion(pred_type_next, next_cancer_type_label) * 0.2
        type_next_loss_r = ce_criterion(pred_type_next_r, next_cancer_type_r_label) * 0.2
        type_next_loss_l = ce_criterion(pred_type_next_l, next_cancer_type_l_label) * 0.2

        # location_before_loss = ce_criterion(location_before, previous_cancer_location_label) * 0.2
        location_next_loss = ce_criterion(location_next, next_cancer_location_label) * 0.2
        location_next_loss_r = ce_criterion(location_next_r, next_cancer_location_r_label) * 0.2
        location_next_loss_l = ce_criterion(location_next_l, next_cancer_location_l_label) * 0.2
        # print('type_next_loss')
        # PCR_loss = ce_criterion(pred_PCR, previous_cancer_PCR_label) * 0.2
        # print('type_next_loss')
        # pT_stage_loss = ce_criterion(pred_pT_stage, previous_cancer_pT_stage_label) * 0.2
        # print('type_next_loss')
        # pN_stage_loss = ce_criterion(pred_pN_stage, previous_cancer_pN_stage_label) * 0.2
        # print('type_next_loss')
        # pM_stage_loss = ce_criterion(pred_pM_stage, previous_cancer_pM_stage_label) * 0.2

        # print('type_next_loss')
        # ER_loss = ce_criterion(pred_ER, previous_cancer_ER_label) * 0.2

        # print('type_next_loss')
        # PR_loss = ce_criterion(pred_PR, previous_cancer_PR_label) * 0.2

        # print('type_next_loss')
        # Her2_loss = ce_criterion(pred_Her2, previous_cancer_Her2_label) * 0.2

        loss = loss + position_next_loss + position_before_loss + type_before_loss + \
               type_next_loss + type_next_loss_r + type_next_loss_l + \
               location_next_loss + location_next_loss_r + location_next_loss_l

    return loss


def train_4views_mtp_220613(model, data_loader, criterion, optimizer, epoch, args):

    model.train()
    adjust_learning_rate(optimizer, epoch, args)
    total_loss, total_top1, total_num, train_bar = 0.0, 0.0, 0, tqdm(data_loader)
    ce_criterion = criterion['ce_criterion']
    l1_criterion = criterion['l1_criterion']
    for input in train_bar:
        imgs = input['imgs'].cuda()
        times = input['times'].cuda()
        lens = input['lens'].cuda()
        labels = input['labels']
        m_labels = input['m_labels']
        risk_label = input['risk'].cuda()

        # tumor_infos = input['tumor_infos']
        m_tumor_infos = input['m_tumor_infos'].cuda()
        # all_risk_label.append(risk_label.cpu().numpy())

        # print('tumor_infos sizee', tumor_infos.size())

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

        pred_risk_label = pred['pred_risk']
        right_side_based_loss = compute_right_side_loss_tumor(args, pred['right_side_based_pred'], m_labels, lens,
                                                              ce_criterion, l1_criterion)

        left_side_based_loss = compute_left_side_loss_tumor(args, pred['left_side_based_pred'], m_labels, lens,
                                                            ce_criterion, l1_criterion)

        exam_based_loss = compute_exam_based_loss_tumor(args, pred['exam_based_pred'], m_labels, lens,
                                                        ce_criterion, l1_criterion)
        if 'side_specific' in args.method:
            final_loss = compute_final_loss_tumor_side_specific(
                args, pred['final_pred'], labels, ce_criterion, l1_criterion)
        else:
            final_loss = compute_final_loss_tumor_0614_(
                args, pred['final_pred'], labels, ce_criterion, l1_criterion)
        loss = right_side_based_loss + left_side_based_loss + exam_based_loss + final_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_loader.batch_size
        total_num += imgs.size(0)
        pred_risk_label = F.softmax(pred_risk_label, dim=1)
        _, risk_predicted = torch.max(pred_risk_label.data, 1)
        total_top1 += (risk_predicted == risk_label).float().sum().item()

        train_bar.set_description(
            'TraEpo:[{}/{}],'
            'lr:{:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr']))

        del imgs
        del _
        del input
        del risk_predicted
        del risk_label
        del pred_risk_label
        gc.collect()

    del train_bar
    gc.collect()

        # train_bar.update(500)
    # print('TraEpo:[{}/{}], lr:{:.4f}, TraLos:{:.2f}, TraAcc:{:.2f}'.format(
    #     epoch, args.epochs, optimizer.param_groups[0]['lr'],
    #     total_loss / total_num, total_top1 / total_num * 100))

    logging.info('TraEpo:[{}/{}], lr:{:.4f}, TraLos:{:.2f}, TraAcc:{:.2f}'.format(
        epoch, args.epochs, optimizer.param_groups[0]['lr'],
        total_loss / total_num, total_top1 / total_num * 100))

    return total_loss / total_num, total_top1 / total_num * 100


def validate_4views_mtp_220613(model, valid_loader, criterion, args, poltroc=False, name='best model'):
    model.eval()
    total_loss, risk_total_top1, total_num_patient = 0.0, 0.0, 0
    all_risk_probabilities, all_risk_predicted, all_risk_label = [], [], []
    all_followups = []
    ce_criterion = criterion['ce_criterion']
    l1_criterion = criterion['l1_criterion']

    with torch.no_grad():
        valid_bar = tqdm(valid_loader)
        # for data, label, _ in valid_bar:
        for input in valid_bar:
            imgs = input['imgs'].cuda()
            times = input['times'].cuda()
            lens = input['lens'].cuda()
            labels = input['labels']
            m_labels = input['m_labels']
            risk_label = input['risk'].cuda()
            # tumor_infos = input['tumor_infos']
            m_tumor_infos = input['m_tumor_infos'].cuda()
            all_risk_label.append(risk_label.cpu().numpy())
            years_last_followup = labels['years_last_followup'].cuda()
            all_followups.append(years_last_followup.cpu().numpy())

            # print('tumor_infos sizee', tumor_infos.size())

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

            pred_risk_label = pred['pred_risk']
            right_side_based_loss = compute_right_side_loss_tumor(args,
                                                                  pred['right_side_based_pred'], m_labels, lens,
                                                                  ce_criterion,
                                                                  l1_criterion)

            left_side_based_loss = compute_left_side_loss_tumor(args,
                                                                pred['left_side_based_pred'], m_labels, lens,
                                                                ce_criterion,
                                                                l1_criterion)

            exam_based_loss = compute_exam_based_loss_tumor(args,
                                                            pred['exam_based_pred'], m_labels, lens, ce_criterion,
                                                            l1_criterion)
            if 'side_specific' in args.method:
                final_loss = compute_final_loss_tumor_side_specific(
                    args, pred['final_pred'], labels, ce_criterion, l1_criterion)
            else:
                final_loss = compute_final_loss_tumor_0614_(
                    args, pred['final_pred'], labels, ce_criterion, l1_criterion)
            loss = right_side_based_loss + left_side_based_loss + exam_based_loss + final_loss

            total_loss += loss.item() * valid_loader.batch_size
            total_num_patient += imgs.size(0)
            pred_risk_label = F.softmax(pred_risk_label, dim=1)
            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            _, risk_predicted = torch.max(pred_risk_label.data, 1)
            all_risk_predicted.append(risk_predicted.cpu().numpy())
            risk_total_top1 += (risk_predicted == risk_label).float().sum().item()

            valid_bar.set_description(
                'Valid Loss: {:.4f}, '
                'risk Acc:{:.2f},%'.format(total_loss / total_num_patient,
                                           risk_total_top1 / total_num_patient * 100))

            del imgs
            del _
            del input
            del risk_predicted
            del risk_label
            del pred_risk_label
            gc.collect()

        del valid_bar
        gc.collect()

            # valid_bar.update(20)


        # confusion_matrix
        risk_cm = comput_cm(all_risk_label, all_risk_predicted)
        # print('risk confusion matrix ', risk_cm)
        logging.info('risk confusion matrix {}'.format(risk_cm))

        # target_names = ['risk_without_5_years', 'risk_within_5_years', 'risk_within_4_years', 'risk_within_3_years',
        #                 'risk_within_2_years', 'risk_within_1_years', 'already_get_cancer']
        target_names = []
        for cl in range(args.num_classes):
            if cl == 0:
                target_names.append('get_BC_after_{:.0f}_years'.format(args.num_classes - cl - 1))
            else:
                target_names.append('get_BC_in_year_{:.0f}'.format(args.num_classes - cl))

        y_true, y_pred = get_y_ture_pred(all_risk_label, all_risk_predicted)
        report = metrics.classification_report(y_true, y_pred, target_names=target_names)

        if name == 'Best Model Val':
            write_report_to_text(args.results_dir_fold, report, 'risk_classification_report.txt')
        # print(report)
        logging.info(report)

        risk_AUC, c_index, decile_recall = polt_roc_curve(
            args, all_risk_label, all_risk_probabilities, all_followups, poltroc, name)

        for cl in range(args.num_classes - 1):
            # print('risk within {:.0f} years AUC is'.format(args.num_classes - cl - 1), risk_AUC[cl])
            logging.info('risk within {:.0f} years AUC is {}'.format((args.num_classes - cl - 1), risk_AUC[cl]))
        logging.info('mean risk is {}'.format(np.mean(risk_AUC)))
        logging.info('c index is {}'.format(c_index))
        logging.info('decile recall is {}'.format(decile_recall))

    # return {
    #     'valid_loss': total_loss / total_num_patient,
    #     'valid_acc': risk_total_top1 / total_num_patient * 100,
    #     'valid_auc': risk_AUC,
    #     'CM': risk_cm,
    # }

    return total_loss / total_num_patient,\
           risk_total_top1 / total_num_patient * 100, risk_AUC, risk_cm


def test_4views_mtp_220613(model, test_loader, criterion, args, poltroc=False, name='best model'):
    model.eval()
    total_loss, risk_total_top1, total_num_patient = 0.0, 0.0, 0
    all_risk_probabilities, all_risk_predicted, all_risk_label = [], [], []
    ce_criterion = criterion['ce_criterion']
    l1_criterion = criterion['l1_criterion']
    all_followups = []

    save_dict = {
        'labels': [],
        'patient_id': [],
        'exam_id': [],
        'times': [],
        'lens': [],
        # 'm_labels': [],
        'risk': [],
        'pred': [],
        # 'm_tumor_infos': [],
        'c_index': [],
        'decile_recall': [],
    }

    with torch.no_grad():
        valid_bar = tqdm(test_loader)
        # for data, label, _ in valid_bar:
        for input in valid_bar:
            save_dict['times'].append(input['times'])
            save_dict['lens'].append(input['lens'])
            save_dict['labels'].append(input['labels'])
            # save_dict['m_labels'].append(input['m_labels'])
            save_dict['risk'].append(input['risk'])
            # save_dict['m_tumor_infos'].append(input['m_tumor_infos'])
            save_dict['patient_id'].append(input['patient_id'])
            save_dict['exam_id'].append(input['exam_id'])
            # input
            imgs = input['imgs'].cuda()
            times = input['times'].cuda()
            lens = input['lens'].cuda()
            labels = input['labels']
            m_labels = input['m_labels']
            risk_label = input['risk'].cuda()
            years_last_followup = labels['years_last_followup'].cuda()
            # tumor_infos = input['tumor_infos']
            m_tumor_infos = input['m_tumor_infos'].cuda()
            all_risk_label.append(risk_label.cpu().numpy())
            all_followups.append(years_last_followup.cpu().numpy())

            # print('m_tumor_infos shape', m_tumor_infos.shape)

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

            save_dict['pred'].append(pred)
            pred_risk_label = pred['pred_risk']
            right_side_based_loss = compute_right_side_loss_tumor(args,
                                                            pred['right_side_based_pred'], m_labels, lens,
                                                            ce_criterion,
                                                            l1_criterion)

            left_side_based_loss = compute_left_side_loss_tumor(args,
                                                          pred['left_side_based_pred'], m_labels, lens, ce_criterion,
                                                          l1_criterion)

            exam_based_loss = compute_exam_based_loss_tumor(args,
                                                      pred['exam_based_pred'], m_labels, lens, ce_criterion,
                                                      l1_criterion)
            if 'side_specific'in args.method:
                final_loss = compute_final_loss_tumor_side_specific(
                    args, pred['final_pred'], labels, ce_criterion, l1_criterion)
            else:
                final_loss = compute_final_loss_tumor_0614_(
                    args, pred['final_pred'], labels, ce_criterion, l1_criterion)

            loss = right_side_based_loss + left_side_based_loss + exam_based_loss + final_loss
            # loss = exam_based_loss + final_loss + right_side_based_loss

            total_loss += loss.item() * test_loader.batch_size
            total_num_patient += imgs.size(0)
            pred_risk_label = F.softmax(pred_risk_label, dim=1)
            all_risk_probabilities.append(pred_risk_label.cpu().numpy())
            _, risk_predicted = torch.max(pred_risk_label.data, 1)
            all_risk_predicted.append(risk_predicted.cpu().numpy())
            risk_total_top1 += (risk_predicted == risk_label).float().sum().item()

            valid_bar.set_description(
                'test Loss: {:.4f},'
                'risk Acc:{:.2f},%'.format(total_loss / total_num_patient,
                                           risk_total_top1 / total_num_patient * 100))

            del imgs
            del _
            del input
            del risk_predicted
            del risk_label
            del pred_risk_label
            del m_tumor_infos
            gc.collect()

        del valid_bar
        gc.collect()

        # # confusion_matrix
        # risk_cm = comput_cm(all_risk_label, all_risk_predicted)
        # print('risk confusion matrix ', risk_cm)
        risk_cm = 'none'

        risk_AUC, c_index, decile_recall = polt_roc_curve_with_CI(
            args, all_risk_label, all_risk_probabilities, all_followups, poltroc, name)

        for cl in range(args.test_num_classes - 1):
            # print('risk within {:.0f} years AUC is'.format(args.test_num_classes - cl - 1), risk_AUC[cl])
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

        save_dict['all_risk_label'] = all_risk_label
        save_dict['all_risk_probabilities'] = all_risk_probabilities
        save_dict['all_risk_predicted'] = all_risk_predicted
        save_dict['risk_AUC'] = risk_AUC
        save_dict['c_index'] = c_index
        save_dict['decile_recall'] = decile_recall

        pickle.dump(save_dict, open('{}/result_{}.pkl'.format(args.results_dir_fold, name), 'wb'))

    return total_loss / total_num_patient,\
           risk_total_top1 / total_num_patient * 100, risk_AUC, risk_cm


def get_train_val_test_demo(model_method):
    # assert model_method in [
    #     '2views', '2views_attention', '4views', '4views_mtp', '4views_mtp_old', '4views_mtp_tumor'], \
    #     "model_method not in ['2views', '4views', '4views_mtp', '4views_mtp_old', '4views_mtp_tumor'], " \
    #     "model method {%s}" % (model_method)

    if model_method == '4views_mtp_tumor' or model_method == 'side_specific_4views_mtp_tumor':
        return train_4views_mtp_220613, validate_4views_mtp_220613, test_4views_mtp_220613
    else:
        raise AttributeError('model method: {}'.format(model_method))
