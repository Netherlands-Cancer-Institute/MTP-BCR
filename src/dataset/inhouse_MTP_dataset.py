import os
import cv2
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import pydicom
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.utils as vutils
import json
import datetime
# image transform
import random
import numbers
from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
from ..utils.utils_robust_custom import imgunit8
# from sklearn.model_selection import KFold, train_test_split


with open('../info/risk_factor.json', 'r') as f:
    inhouse_dict = json.load(f)


with open('../info/tumor_info_label.json', 'r') as f:
    tumor_info_label_dict = json.load(f)


def data_info_clean(data_info, agrs):
    """
    1. clean exams with missing image
    2. clean exams with abnormal finding (bi-Rads > 2)
    3. clean exams without enough followup
    """

    if agrs.missing_clean:
        data_info['PATH_L_CC_processed'][data_info['PATH_L_CC_processed'].isna()] = 'none'
        data_info['PATH_R_CC_processed'][data_info['PATH_R_CC_processed'].isna()] = 'none'
        data_info['PATH_L_MLO_processed'][data_info['PATH_L_MLO_processed'].isna()] = 'none'
        data_info['PATH_R_MLO_processed'][data_info['PATH_R_MLO_processed'].isna()] = 'none'

        data_info = data_info[(data_info['PATH_L_CC_processed'] != 'none')
                              & (data_info['PATH_R_CC_processed'] != 'none')
                              & (data_info['PATH_L_MLO_processed'] != 'none')
                              & (data_info['PATH_R_MLO_processed'] != 'none')]

    if agrs.biopsy_clean:
        data_info['birads'][data_info['birads'].isna()] = -1
        data_info['clinical_group'] = 'screening'
        data_info['next_cancer_date'][data_info['next_cancer_date'].isna()] = 'none'
        for i in range(len(data_info)):
            diagnosis_date = data_info['next_cancer_date'].iloc[i]
            if diagnosis_date != 'none' and data_info["birads"].iloc[i] != 1 and \
                    data_info["birads"].iloc[i] != 2:
                diagnosis_date_ = datetime.datetime.strptime(str(diagnosis_date), '%Y/%m/%d')
                case_exam_data = data_info['exam_date'].iloc[i]
                case_exam_data_ = datetime.datetime.strptime(str(case_exam_data), '%Y/%m/%d')
                gap = round((diagnosis_date_ - case_exam_data_).days)
                if gap <= 90:
                    data_info['clinical_group'].iloc[i] = 'diagnosis'
        data_info = data_info[data_info['clinical_group'] == 'screening']

    if agrs.birads_clean:
        # data_info['birads'][data_info['birads'].isna()] = -1
        data_info = data_info[(data_info['birads'] == 0)
                              | (data_info['birads'] == 1)
                              | (data_info['birads'] == 2)]
                              # | (data_info['birads'] == -1)]


    if agrs.years_at_least_followup != 0:
        x = agrs.years_at_least_followup
        data_info = data_info[(data_info['years_to_last_followup'] > x - 1)
                              | (data_info["years_to_next_cancer"] != 100)]

    return data_info


def data_info_clean_hist(data_info, agrs):
    """
    1. clean exams with missing image
    2. clean exams with abnormal finding (bi-Rads > 2)
    3. clean exams without enough followup
    """

    if agrs.missing_clean:
        data_info['PATH_L_CC_processed'][data_info['PATH_L_CC_processed'].isna()] = 'none'
        data_info['PATH_R_CC_processed'][data_info['PATH_R_CC_processed'].isna()] = 'none'
        data_info['PATH_L_MLO_processed'][data_info['PATH_L_MLO_processed'].isna()] = 'none'
        data_info['PATH_R_MLO_processed'][data_info['PATH_R_MLO_processed'].isna()] = 'none'

        data_info = data_info[(data_info['PATH_L_CC_processed'] != 'none')
                              & (data_info['PATH_R_CC_processed'] != 'none')
                              & (data_info['PATH_L_MLO_processed'] != 'none')
                              & (data_info['PATH_R_MLO_processed'] != 'none')]

    if agrs.years_at_least_followup != 0:
        x = agrs.years_at_least_followup
        data_info = data_info[(data_info['years_to_last_followup'] > x - 1)
                              | (data_info["years_to_next_cancer"] != 100)]

    return data_info


def get_tumor_label(pid, tumorid):
    if 'unknow' not in tumorid and ',' not in tumorid and tumorid != 'nan':
        tumorid = str(float(tumorid))

    tumor_labels = []

    if ',' in tumorid and 'unknow' in tumorid:
        tumorid = tumorid.replace('unknow', '')
        tumorid = tumorid.replace(',', '')
    else:
        if ',' in tumorid:
            tumorid = tumorid.split(",")[0]


    lists = [
        'TRTU_Klinische_T',
        'TRTU_Klinische_N',
        'TRTU_Klinische_M',
        'TRTU_Post_chir_T',
        'TRTU_Post_chir_N',
        'TRTU_Post_chir_M',
        'TRTU_Aantal_onderzochte_lymfklieren',
        'TRTU_Aantal_positieve_lymfklieren',
        'Axillary dissection',
        'TRTU_Sentinel_node_procedure_code',
        # 'TRTU_Sentinel_node_procedure_oms',
        'TRTU_Soort_mamma_ok_code',
        # 'TRTU_Soort_mamma_ok_oms',
        'TRTU_Risicofactor_1_code',
        # 'TRTU_Risicofactor_1_oms',
        'TRTU_Risicofactor_2_code',
        # 'TRTU_Risicofactor_2_oms',
        'TRTU_Risicofactor_3_code',
        # 'TRTU_Risicofactor_3_oms',
        'TRTU_Chirurgie_gegeven_ind',
        'TRTU_Radiotherapie_gegeven_ind',
        'TRTU_Chemotherapie_gegeven_ind',
        'TRTU_Hormonale_therapie_gegeven_ind',
        'TRTU_Immunotherapie_gegeven_ind',
        'TRTU_Neo_adjuvante_radiotherapie_ind',
        'TRTU_Neo_adjuvante_chemotherapie_ind',
        'TRTU_Neo_adjuvante_hormonale_therapie_ind',
        'TRTU_Neo_adjuvante_immunotherapie_ind',
        'TRTU_Neo_adjuvante_therapie_ind',
        'TRTU_Adjuvante_radiotherapie_ind',
        'TRTU_Adjuvante_chemotherapie_ind',
        'TRTU_Adjuvante_hormonale_therapie_ind',
        'TRTU_Adjuvante_immunotherapie_ind',
        'TRTU_Adjuvante_therapie_ind',
    ]
    if tumorid != 'nan' and tumorid in inhouse_dict[pid]['tumor']:
        tumpr_dict = inhouse_dict[pid]['tumor'][tumorid]
        for list in lists:
            if list in tumor_info_label_dict and list in tumpr_dict:
                key = str(tumpr_dict[list])
                label = tumor_info_label_dict[list][key]+1
                tumor_labels.append(label)
            else:
                tumor_labels.append(-1)
    else:
        for list in lists:
            tumor_labels.append(0)

    return tumor_labels,


def get_history_test_data_info(args):
    data_info = pd.read_csv(args.reuse_csv + '/history_data_info.csv')
    data_info = data_info_clean_hist(data_info, args)
    data_info = data_info_replace(data_info)
    data_info = data_info_get_label(data_info, args)
    return data_info


def data_info_replace(csv_data_info):

    csv_data_info['years_last_followup'] = csv_data_info['years_to_last_followup']

    for i in range(len(csv_data_info)):
        if 'unknow' in str(csv_data_info["next_cancer_location"].iloc[i]) \
                or 'none' in str(csv_data_info["next_cancer_location"].iloc[i]) \
                or ',' in str(csv_data_info["next_cancer_location"].iloc[i]):
            csv_data_info["next_cancer_location"].iloc[i] = -1
        if 'unknow' in str(csv_data_info["previous_cancer_location"].iloc[i]) \
                or 'none' in str(csv_data_info["previous_cancer_location"].iloc[i]) \
                or ',' in str(csv_data_info["previous_cancer_location"].iloc[i]):
            csv_data_info["previous_cancer_location"].iloc[i] = -1

        if csv_data_info["years_to_next_cancer"].iloc[i] != 100:
            csv_data_info["years_last_followup"].iloc[i] = 100

    # csv_data_info['next_cancer_location']['unknow' in csv_data_info['next_cancer_location']] = -1
    # csv_data_info['next_cancer_location']['none' in csv_data_info['next_cancer_location']] = -1
    # csv_data_info['next_cancer_location'][',' in csv_data_info['next_cancer_location']] = -1

    csv_data_info['age_ig'] = 1

    csv_data_info["age"][csv_data_info["age"] == 'none'] = -1
    csv_data_info['age_ig'][csv_data_info["age"] == -1] = 0

    csv_data_info['birads'][csv_data_info['birads'].isna()] = -1
    csv_data_info['density'][csv_data_info['density'].isna()] = -1
    csv_data_info['density'][csv_data_info['density'] == 1] = 0
    csv_data_info['density'][csv_data_info['density'] == 2] = 1
    csv_data_info['density'][csv_data_info['density'] == 3] = 2
    csv_data_info['density'][csv_data_info['density'] == 4] = 3
    # race
    csv_data_info['race'][csv_data_info['race'] == 'unknow'] = -1
    csv_data_info['race'][csv_data_info['race'] == 'white'] = 0
    csv_data_info['race'][csv_data_info['race'] == 'asian'] = 1
    csv_data_info['race'][csv_data_info['race'] == 'black'] = 2
    csv_data_info['race'][csv_data_info['race'] == 'other'] = 3
    # gene
    csv_data_info['gene'][csv_data_info['gene'] == 'unknow'] = -1
    csv_data_info['gene'][csv_data_info['gene'] == 'none'] = 0
    csv_data_info['gene'][csv_data_info['gene'] == 'brca1'] = 1
    csv_data_info['gene'][csv_data_info['gene'] == 'brca2'] = 2
    csv_data_info['gene'][csv_data_info['gene'] == 'chek2'] = 3
    csv_data_info['gene'][csv_data_info['gene'] == 'tp53'] = 4
    # menarche
    csv_data_info['menarche'][csv_data_info['menarche'] == 'unknow'] = -1
    csv_data_info['menarche'][csv_data_info['menarche'] == 115] = 15
    # menopausal
    csv_data_info['menopausal'][csv_data_info['menopausal'] == 'unknow'] = -1
    csv_data_info['menopausal'][csv_data_info['menopausal'] == 'no'] = 0
    csv_data_info['menopausal'][csv_data_info['menopausal'] == 'yes'] = 1
    # ovarian_cancer
    csv_data_info['ovarian_cancer'][csv_data_info['ovarian_cancer'] == 'no'] = 0
    csv_data_info['ovarian_cancer'][csv_data_info['ovarian_cancer'] == 'yes'] = 1
    # family_history
    csv_data_info['family_history'][csv_data_info['family_history'] == 'negative'] = 0
    csv_data_info['family_history'][csv_data_info['family_history'] == 'positive'] = 1
    csv_data_info['family_history_breast'][csv_data_info['family_history_breast'] == 'negative'] = 0
    csv_data_info['family_history_breast'][csv_data_info['family_history_breast'] == 'positive'] = 1
    csv_data_info['family_history_other'][csv_data_info['family_history_other'] == 'negative'] = 0
    csv_data_info['family_history_other'][csv_data_info['family_history_other'] == 'positive'] = 1
    csv_data_info['family_history_ovarian'][csv_data_info['family_history_ovarian'] == 'negative'] = 0
    csv_data_info['family_history_ovarian'][csv_data_info['family_history_ovarian'] == 'positive'] = 1
    # ManufacturerModelName
    csv_data_info['ManufacturerModelName'][csv_data_info['ManufacturerModelName'] == 'none'] = -1
    csv_data_info['ManufacturerModelName'][csv_data_info['ManufacturerModelName'] == 'Selenia Dimensions'] = 0
    csv_data_info['ManufacturerModelName'][csv_data_info['ManufacturerModelName'] == 'Lorad Selenia'] = 1
    csv_data_info['ManufacturerModelName'][csv_data_info['ManufacturerModelName'] == 'Hologic Selenia'] = 2

    # next_cancer_position
    csv_data_info['next_cancer_position'][csv_data_info['next_cancer_position'].isna()] = -1
    csv_data_info['next_cancer_position'][csv_data_info['next_cancer_position'] == 'unknow'] = -1
    csv_data_info['next_cancer_position'][csv_data_info['next_cancer_position'] == 'both'] = 0
    csv_data_info['next_cancer_position'][csv_data_info['next_cancer_position'] == 'left'] = 1
    csv_data_info['next_cancer_position'][csv_data_info['next_cancer_position'] == 'right'] = 2
    # next_cancer_type
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'].isna()] = -1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'unknow'] = -1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'primare'] = 1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'recidief'] = 2
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'primare,recidief'] = 0
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'recidief,primare'] = 0
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'primare,unknow'] = -1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'recidief,unknow'] = -1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'unknow,primare'] = -1
    csv_data_info['next_cancer_type'][csv_data_info['next_cancer_type'] == 'unknow,recidief'] = -1

    # next_cancer_location
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'].isna()] = -1
    # csv_data_info['next_cancer_location']['unknow' in csv_data_info['next_cancer_location']] = -1
    # csv_data_info['next_cancer_location']['none' in csv_data_info['next_cancer_location']] = -1
    # csv_data_info['next_cancer_location'][',' in csv_data_info['next_cancer_location']] = -1
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.0'] = 0
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.1'] = 1
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.2'] = 2
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.3'] = 3
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.4'] = 4
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.5'] = 5
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.6'] = 6
    # csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.7'] = 7
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.8'] = 7
    csv_data_info['next_cancer_location'][csv_data_info['next_cancer_location'] == 'C50.9'] = -1

    # previous_cancer_position
    csv_data_info['previous_cancer_position'][csv_data_info['previous_cancer_position'].isna()] = 0
    csv_data_info['previous_cancer_position'][csv_data_info['previous_cancer_position'] == 'unknow'] = -1
    csv_data_info['previous_cancer_position'][csv_data_info['previous_cancer_position'] == 'both'] = 1
    csv_data_info['previous_cancer_position'][csv_data_info['previous_cancer_position'] == 'left'] = 2
    csv_data_info['previous_cancer_position'][csv_data_info['previous_cancer_position'] == 'right'] = 3
    # previous_cancer_type
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'].isna()] = -1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'unknow'] = -1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'primare'] = 1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'recidief'] = 2
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'primare,recidief'] = 0
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'recidief,primare'] = 0
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'primare,unknow'] = -1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'recidief,unknow'] = -1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'unknow,primare'] = -1
    csv_data_info['previous_cancer_type'][csv_data_info['previous_cancer_type'] == 'unknow,recidief'] = -1

    # previous_cancer_location
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'].isna()] = -1
    # csv_data_info['previous_cancer_location']['unknow' in csv_data_info['previous_cancer_location']] = -1
    # csv_data_info['previous_cancer_location']['none' in csv_data_info['previous_cancer_location']] = -1
    # csv_data_info['previous_cancer_location'][',' in csv_data_info['previous_cancer_location']] = -1
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.0'] = 0
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.1'] = 1
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.2'] = 2
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.3'] = 3
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.4'] = 4
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.5'] = 5
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.6'] = 6
    # csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.7'] = 7
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.8'] = 7
    csv_data_info['previous_cancer_location'][csv_data_info['previous_cancer_location'] == 'C50.9'] = -1

    # next_cancer_type_left
    csv_data_info['next_cancer_type_left'][csv_data_info['next_cancer_type_left'].isna()] = -1
    csv_data_info['next_cancer_type_left'][csv_data_info['next_cancer_type_left'] == 'unknow'] = -1
    csv_data_info['next_cancer_type_left'][csv_data_info['next_cancer_type_left'] == 'primare'] = 0
    csv_data_info['next_cancer_type_left'][csv_data_info['next_cancer_type_left'] == 'recidief'] = 1

    # previous_cancer_type_left
    csv_data_info['previous_cancer_type_left'][csv_data_info['previous_cancer_type_left'].isna()] = -1
    csv_data_info['previous_cancer_type_left'][csv_data_info['previous_cancer_type_left'] == 'unknow'] = -1
    csv_data_info['previous_cancer_type_left'][csv_data_info['previous_cancer_type_left'] == 'primare'] = 0
    csv_data_info['previous_cancer_type_left'][csv_data_info['previous_cancer_type_left'] == 'recidief'] = 1

    # next_cancer_location_left
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'].isna()] = -1
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'unknow'] = -1
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'none'] = -1
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.0'] = 0
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.1'] = 1
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.2'] = 2
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.3'] = 3
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.4'] = 4
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.5'] = 5
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.6'] = 6
    # csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.7'] = 7
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.8'] = 7
    csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'] == 'C50.9'] = -1
    # csv_data_info['next_cancer_location_left'][csv_data_info['next_cancer_location_left'].isna()] = -1

    # previous_cancer_location_left
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'].isna()] = -1
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'unknow'] = -1
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'none'] = -1
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.0'] = 0
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.1'] = 1
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.2'] = 2
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.3'] = 3
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.4'] = 4
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.5'] = 5
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.6'] = 6
    # csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.7'] = 7
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.8'] = 7
    csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'] == 'C50.9'] = -1
    # csv_data_info['previous_cancer_location_left'][csv_data_info['previous_cancer_location_left'].isnumeric()] = -1

    # next_cancer_type_right
    csv_data_info['next_cancer_type_right'][csv_data_info['next_cancer_type_right'].isna()] = -1
    csv_data_info['next_cancer_type_right'][csv_data_info['next_cancer_type_right'] == 'unknow'] = -1
    csv_data_info['next_cancer_type_right'][csv_data_info['next_cancer_type_right'] == 'primare'] = 0
    csv_data_info['next_cancer_type_right'][csv_data_info['next_cancer_type_right'] == 'recidief'] = 1

    # previous_cancer_type_right
    csv_data_info['previous_cancer_type_right'][csv_data_info['previous_cancer_type_right'].isna()] = -1
    csv_data_info['previous_cancer_type_right'][csv_data_info['previous_cancer_type_right'] == 'unknow'] = -1
    csv_data_info['previous_cancer_type_right'][csv_data_info['previous_cancer_type_right'] == 'primare'] = 0
    csv_data_info['previous_cancer_type_right'][csv_data_info['previous_cancer_type_right'] == 'recidief'] = 1

    # next_cancer_location_right
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'].isna()] = -1
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'unknow'] = -1
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'none'] = -1
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.0'] = 0
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.1'] = 1
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.2'] = 2
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.3'] = 3
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.4'] = 4
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.5'] = 5
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.6'] = 6
    # csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.7'] = 7
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.8'] = 7
    csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'] == 'C50.9'] = -1
    # csv_data_info['next_cancer_location_right'][csv_data_info['next_cancer_location_right'].isna()] = -1

    # previous_cancer_location_right
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'].isna()] = -1
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'unknow'] = -1
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'none'] = -1
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.0'] = 0
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.1'] = 1
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.2'] = 2
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.3'] = 3
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.4'] = 4
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.5'] = 5
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.6'] = 6
    # csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.7'] = 7
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.8'] = 7
    csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'] == 'C50.9'] = -1
    # csv_data_info['previous_cancer_location_right'][csv_data_info['previous_cancer_location_right'].isna()] = -1
    return csv_data_info


def data_info_get_label(data_info, args):
    """
    risk, risk_r, risk_l
    history, history_r, history_l
    # next_cancer_position, previous_cancer_position
    # next_cancer_type, next_cancer_type_left, next_cancer_type_right,
    # previous_cancer_type, previous_cancer_type_left, previous_cancer_type_left,
    # previous_cancer_location_left, previous_cancer_location_right
    # age,
    # birads,
    # density,
    # race,
    # gene,
    # menarche,
    # menopausal,
    # ovarian_cancer,
    # family_history, family_history_breast, family_history_other, family_history_ovarian,
    # ManufacturerModelName
    """

    # risk
    num_classes = args.num_classes
    years_to_next_cancer = np.asarray(data_info["years_to_next_cancer"], dtype='int64')
    risks = num_classes - (years_to_next_cancer + 1)
    risks[risks < 0] = 0
    # risk_r
    years_to_next_cancer_right = np.asarray(data_info["years_to_next_cancer_right"], dtype='int64')
    risks_r = num_classes - (years_to_next_cancer_right + 1)
    risks_r[risks_r < 0] = 0
    # risk_l
    years_to_next_cancer_left = np.asarray(data_info["years_to_next_cancer_left"], dtype='int64')
    risks_l = num_classes - (years_to_next_cancer_left + 1)
    risks_l[risks_l < 0] = 0

    # history
    years_from_previous_cancer = np.asarray(data_info['years_from_previous_cancer'], dtype='int64')
    history = args.years_of_history - (years_from_previous_cancer + 1)
    history[history < 0] = 0

    # history
    years_from_previous_cancer_right = np.asarray(data_info['years_from_previous_cancer_right'], dtype='int64')
    history_r = args.years_of_history - (years_from_previous_cancer_right + 1)
    history_r[history_r < 0] = 0

    # history
    years_from_previous_cancer_left = np.asarray(data_info['years_from_previous_cancer_left'], dtype='int64')
    history_l = args.years_of_history - (years_from_previous_cancer_left + 1)
    history_l[history_l < 0] = 0

    # labels
    labels = num_classes - (years_to_next_cancer + 1)
    labels[labels <= (args.num_classes - args.test_num_classes)] = 0
    labels[labels > 0] = 1

    data_info['risks'] = risks
    data_info['risks_r'] = risks_r
    data_info['risks_l'] = risks_l
    data_info['history'] = history
    data_info['history_r'] = history_r
    data_info['history_l'] = history_l
    data_info['labels'] = labels

    # data_info['years_last_followup'] = np.asarray(data_info['years_from_previous_cancer_left'], dtype='int64')
    # data_info['years_last_followup'][data_info["years_to_next_cancer"] != 100] = 1000

    # data_info['next_cancer_position'][data_info["labels"] == 0] = -1
    # return data_info, labels, risks, risks_r, risks_l, history, history_r, history_l
    return data_info


class inhouse_Dataset(Dataset):
    def __init__(self, args, data_info, file_path, transform, downsample=False):
        """
        downsample: Not supported yet, used to speed up the training process by randomly
                    selecting part of the training set in a ratio, e.g: ratio of 0.1 .
        """

        self.img_size = args.img_size
        self.file_path = file_path

        data_info = data_info_clean(data_info, args)
        data_info = data_info_replace(data_info)
        data_info = data_info_get_label(data_info, args)
        self.data_info = data_info
        risks = data_info['risks']
        self.risks = np.asarray(risks, dtype='int64')
        labels = data_info['labels']
        self.labels = np.asarray(labels, dtype='int64')
        self.num_time_points = args.num_time_points

        image_file_path = np.vstack((
            np.asarray(data_info.PATH_L_CC_processed),
            np.asarray(data_info.PATH_L_MLO_processed),
            np.asarray(data_info.PATH_R_CC_processed),
            np.asarray(data_info.PATH_R_MLO_processed)
             ))
        self.img_file_path = image_file_path
        self.transform = transform
        self.history_selct_method = args.history_selct_method
        self.history_test_data_info = get_history_test_data_info(args)

    def __getitem__(self, index):
        # path = self.img_file_path[:, index]
        risk = self.risks[index]
        patient_id = self.data_info['patient_id'].iloc[index]
        exam_id = self.data_info['exam_id'].iloc[index]
        exam_date = self.data_info['exam_date'].iloc[index]
        labels = self.__get_label(self.data_info, index)
        history_data_info, num_exam = self.__get_history(patient_id, exam_id, exam_date)

        tumor_infos = get_tumor_label(
                    str(self.data_info['patient_id'].iloc[index]).zfill(8),
                    str(self.data_info['previous_cancer_id'].iloc[index]))
        tumor_infos = torch.from_numpy(np.asarray(tumor_infos, dtype='float32'))

        imgs = []
        times = []
        m_tumor_infos = []

        label_lib = ['risks', 'risks_r', 'risks_l',
                     'history', 'history_r', 'history_l',
                     'next_cancer_position', 'previous_cancer_position',
                     'next_cancer_type', 'next_cancer_type_left', 'next_cancer_type_right',
                     'previous_cancer_type', 'previous_cancer_type_left', 'previous_cancer_type_right',
                     'previous_cancer_location', 'previous_cancer_location_left',  'previous_cancer_location_right',
                     'next_cancer_location', 'next_cancer_location_left',  'next_cancer_location_right',
                     'birads', 'density', 'race', 'gene', 'menarche', 'menopausal', 'ovarian_cancer',
                     'family_history', 'family_history_breast', 'family_history_other', 'family_history_ovarian',
                     'ManufacturerModelName', 'age', 'age_ig',
                     'previous_cancer_PCR',
                     'previous_cancer_pT_stage',
                     'previous_cancer_pN_stage',
                     'previous_cancer_pM_stage',
                     'previous_cancer_ER',
                     'previous_cancer_PR',
                     'previous_cancer_Her2',
                     'previous_cancer_PCR_right',
                     'previous_cancer_pT_stage_right',
                     'previous_cancer_pN_stage_right',
                     'previous_cancer_pM_stage_right',
                     'previous_cancer_ER_right',
                     'previous_cancer_PR_right',
                     'previous_cancer_Her2_right',
                     'previous_cancer_PCR_left',
                     'previous_cancer_pT_stage_left',
                     'previous_cancer_pN_stage_left',
                     'previous_cancer_pM_stage_left',
                     'previous_cancer_ER_left',
                     'previous_cancer_PR_left',
                     'previous_cancer_Her2_left',
                     'years_last_followup',
                     ]

        m_labels = {}
        for label_ in label_lib:
            m_labels[label_] = []

        for i in range(self.num_time_points):
            if i+1 <= num_exam:
                case = torch.ones(4, 1, *self.img_size) * -1

                PATH_R_CC_processed = str(history_data_info['PATH_R_CC_processed'].iloc[i])
                PATH_R_MLO_processed = str(history_data_info['PATH_R_MLO_processed'].iloc[i])
                PATH_L_CC_processed = str(history_data_info['PATH_L_CC_processed'].iloc[i])
                PATH_L_MLO_processed = str(history_data_info['PATH_L_MLO_processed'].iloc[i])

                if 'home' in PATH_R_CC_processed:
                    img_R_CC = self.__getimg(self.file_path, PATH_R_CC_processed)
                    if img_R_CC is not None:
                        case[0, :, :, :] = img_R_CC

                if 'home' in PATH_R_MLO_processed:
                    img_R_MLO = self.__getimg(self.file_path, PATH_R_MLO_processed)
                    if img_R_MLO is not None:
                        case[1, :, :, :] = img_R_MLO

                if 'home' in PATH_L_CC_processed:
                    img_L_CC = self.__getimg(self.file_path, PATH_L_CC_processed)
                    if img_L_CC is not None:
                        case[2, :, :, :] = img_L_CC

                if 'home' in PATH_L_MLO_processed:
                    img_L_MLO = self.__getimg(self.file_path, PATH_L_MLO_processed)
                    if img_L_MLO is not None:
                        case[3, :, :, :] = img_L_MLO
                # case[0, :, :, :] = self.__getimg(self.file_path, history_data_info['PATH_R_CC_processed'].iloc[i])
                # case[1, :, :, :] = self.__getimg(self.file_path, history_data_info['PATH_R_MLO_processed'].iloc[i])
                # case[2, :, :, :] = self.__getimg(self.file_path, history_data_info['PATH_L_CC_processed'].iloc[i])
                # case[3, :, :, :] = self.__getimg(self.file_path, history_data_info['PATH_L_MLO_processed'].iloc[i])
                imgs.append(case)

                tumor_info = torch.ones(2, 29) * -1
                if 'home' in PATH_R_CC_processed or 'home' in PATH_R_MLO_processed:
                    tumor_info[0,:] = torch.from_numpy(np.asarray(get_tumor_label(
                        str(history_data_info['patient_id'].iloc[i]).zfill(8),
                        str(history_data_info['previous_cancer_id_right'].iloc[i])), dtype='float32'))

                if 'home' in PATH_L_CC_processed or 'home' in PATH_L_MLO_processed:
                    tumor_info[1, :] = torch.from_numpy(np.asarray(get_tumor_label(
                        str(history_data_info['patient_id'].iloc[i]).zfill(8),
                        str(history_data_info['previous_cancer_id_left'].iloc[i])), dtype='float32'))
                    m_tumor_infos.append(torch.from_numpy(np.asarray(tumor_info, dtype='float32')))

                case_exam_data = history_data_info['exam_date'].iloc[i]
                exam_date_ = datetime.datetime.strptime(str(exam_date), '%Y/%m/%d')
                # exam_date_ = datetime.datetime.strptime(str(exam_date), '%Y-%m-%d')
                # case_exam_data_ = datetime.datetime.strptime(str(case_exam_data), '%Y-%m-%d')
                case_exam_data_ = datetime.datetime.strptime(str(case_exam_data), '%Y/%m/%d')
                gap = round((exam_date_ - case_exam_data_).days / 30)
                times.append(gap)

                labels_ = self.__get_label(history_data_info, i)
                for label_ in label_lib:
                    m_labels[label_].append(labels_[label_])

            else:
                for label_ in label_lib:
                    if label_ != 'age':
                        m_labels[label_].append(torch.from_numpy(np.asarray(-1, dtype='int64')))
                    else:
                        m_labels[label_].append(torch.from_numpy(np.asarray(-1, dtype='float32')))
                imgs.append(torch.zeros(4, 1, *self.img_size))
                times.append(-1)
                m_tumor_infos.append(torch.ones(2, 29) * -1)

        # age = self.age[index]
        # # print(age)
        # age = np.asarray(age / 100, dtype='float32')
        # birads = self.birads[index]
        # previous_cancer = self.previous_cancers[index]
        times = torch.as_tensor(times)
        imgs = torch.stack(imgs)

        for label_ in label_lib:
            m_labels[label_] = torch.stack(m_labels[label_])

        m_tumor_infos = torch.stack(m_tumor_infos)

        return {'imgs': imgs,
                'times': times,
                'lens': num_exam,
                'risk': risk,
                'labels': labels,
                'tumor_infos': tumor_infos,
                'm_labels': m_labels,
                'm_tumor_infos': m_tumor_infos,
                'patient_id': patient_id,
                'exam_id': exam_id,
                'view': [np.asarray(0, dtype='int64'),
                         np.asarray(1, dtype='int64'),
                         np.asarray(0, dtype='int64'),
                         np.asarray(1, dtype='int64')
                         ],
                # 'img_name': [path[0], path[1]]
                }

    def __len__(self):
        return len(self.data_info)

    def __getimg(self, file_path, path):
        img_path = path

        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if image is not None:
            if "R_" in img_path:
                image = cv2.flip(image, 1)
            image = imgunit8(image)
            img = Image.fromarray(image)
            if self.transform is not None:
                img = self.transform(img)
        else:
            logging.info('{} wrong!!!'.format(img_path))
            img = None
        return img

    def __get_history(self, patient_id, exam_id, exam_date):
        # csv_data_info = self.data_info
        csv_data_info = self.history_test_data_info
        # index_ = [i for (i, v) in enumerate(list(csv_data_info["patient_id"])) if v == patient_id]
        # patient_exam_history_ = csv_data_info.iloc[index_, :]
        patient_exam_history_ = csv_data_info[csv_data_info["patient_id"] == patient_id]

        index_ = [i for (i, v) in enumerate(list(patient_exam_history_["exam_date"])) if v <= exam_date]
        history_data_info = patient_exam_history_.iloc[index_, :]
        history_data_info = history_data_info.sort_values(by=['exam_date'], ascending=True)

        # print('There are {} history exams before the target exam'.format(len(patient_exam_history) - 1))

        num_exam = len(history_data_info)

        if num_exam > self.num_time_points:
            if self.history_selct_method == 'last':
                # select the last few history exams
                history_data_info = history_data_info.iloc[num_exam-self.num_time_points:num_exam, :]
            else:
                # select randomly from the history exams
                target_data_info = history_data_info.iloc[num_exam - 1:num_exam, :]
                history_data_info_ = history_data_info.iloc[0:num_exam - 2, :]
                select_history_data_info_ = history_data_info_.sample(n=(self.num_time_points - 1))
                history_data_info = pd.concat([select_history_data_info_, target_data_info])

        history_data_info = history_data_info.sort_values(by=['exam_date'], ascending=True)

        return history_data_info, len(history_data_info)

    def __get_label(self, data_info, index):
        # get label
        # ______________________________________________________________
        labels = {}
        label_lib = ['risks', 'risks_r', 'risks_l',
                     'history', 'history_r', 'history_l',
                     'next_cancer_position', 'previous_cancer_position',
                     'next_cancer_type', 'next_cancer_type_left', 'next_cancer_type_right',
                     'previous_cancer_type', 'previous_cancer_type_left', 'previous_cancer_type_right',
                     'previous_cancer_location', 'previous_cancer_location_left',  'previous_cancer_location_right',
                     'next_cancer_location', 'next_cancer_location_left',  'next_cancer_location_right',
                     'birads', 'density', 'race', 'gene', 'menarche', 'menopausal', 'ovarian_cancer',
                     'family_history', 'family_history_breast', 'family_history_other', 'family_history_ovarian',
                     'ManufacturerModelName', 'age', 'age_ig',
                     'previous_cancer_PCR',
                     'previous_cancer_pT_stage',
                     'previous_cancer_pN_stage',
                     'previous_cancer_pM_stage',
                     'previous_cancer_ER',
                     'previous_cancer_PR',
                     'previous_cancer_Her2',
                     'previous_cancer_PCR_right',
                     'previous_cancer_pT_stage_right',
                     'previous_cancer_pN_stage_right',
                     'previous_cancer_pM_stage_right',
                     'previous_cancer_ER_right',
                     'previous_cancer_PR_right',
                     'previous_cancer_Her2_right',
                     'previous_cancer_PCR_left',
                     'previous_cancer_pT_stage_left',
                     'previous_cancer_pN_stage_left',
                     'previous_cancer_pM_stage_left',
                     'previous_cancer_ER_left',
                     'previous_cancer_PR_left',
                     'previous_cancer_Her2_left',
                     'years_last_followup',
                     ]

        for label_ in label_lib:
            # print(label_)

            label = str(data_info[label_].iloc[index])
            if label == 'nan':
                label = -1
            else:
                label = data_info[label_].iloc[index]

            if label_ == 'age':
                labels[label_] = torch.from_numpy(np.asarray(label, dtype='float32'))
            else:
                labels[label_] = torch.from_numpy(np.asarray(label, dtype='int64'))

            if 'right' in label_ or 'risks_r' in label_ or 'history_r' in label_:
                PATH_R_CC_processed = str(data_info['PATH_R_CC_processed'].iloc[index])
                PATH_R_MLO_processed = str(data_info['PATH_R_MLO_processed'].iloc[index])
                if 'home' not in PATH_R_CC_processed and 'home' not in PATH_R_MLO_processed:
                    labels[label_] = torch.from_numpy(np.asarray(-1, dtype='int64'))
            elif 'left' in label_ or 'risks_l' in label_ or 'history_l' in label_:
                PATH_L_CC_processed = str(data_info['PATH_L_CC_processed'].iloc[index])
                PATH_L_MLO_processed = str(data_info['PATH_L_MLO_processed'].iloc[index])
                if 'home' not in PATH_L_CC_processed and 'home' not in PATH_L_MLO_processed:
                    labels[label_] = torch.from_numpy(np.asarray(-1, dtype='int64'))

        # ______________________________________________________________
        return labels
