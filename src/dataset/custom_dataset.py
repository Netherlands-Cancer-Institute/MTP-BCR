import os
import cv2
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import datetime
from PIL import Image
from ..utils.utils_robust_custom import imgunit8
import random
random.seed(10)
# print(random.randint(0, 9))

"""
External (Custom) Dataset for Single Time Point（STP）Examination Mammogram (R-CC, R-MLO, L-CC, L-MLO)
*************************************************************************
******* NOTE: Just Support the Custom Single Time Point Exam Now. *******
******* NOTE: Risk factors are not supported now.                 *******
*************************************************************************
CSV formate:
--------> Patient_id, exam_id, laterality, view, years_to_cancer, years_to_last_followup, split_group,
--------> PATH_L_CC, PATH_R_CC, PATH_L_MLO, PATH_R_MLO

Other labels in the CSV: (Our MTP-BCR model required for breast level risk prediction):
--------> years_to_cancer_right, years_to_cancer_left, exam_date
"""


def get_id_list(
        data_info,
        name='exam_id'  # patient_id or exam_id
):
    all_id = list(data_info[name])
    _id = []
    [_id.append(i) for i in all_id if not i in _id]
    return _id


class Custom_risk_stp_Dataset(Dataset):  # Single time point dataset
    def __init__(self, args, data_info, transform, risk_factor_data_info=None):
        self.img_size = args.img_size
        self.data_info = data_info
        self.history_data_info = data_info
        exam_ids = get_id_list(data_info)
        self.exam_ids = exam_ids
        self.num_time_points = args.num_time_points
        self.transform = transform
        self.file_folder = args.image_dir
        image_file_path = np.vstack(
            (np.asarray(data_info.PATH_R_CC),
             np.asarray(data_info.PATH_R_MLO),
             np.asarray(data_info.PATH_L_CC),
             np.asarray(data_info.PATH_L_MLO),
             ))
        self.img_file_path = image_file_path
        # self.risk_factor_data_info = risk_factor_data_info

    def __getitem__(self, index):
        exam_id = self.exam_ids[index]
        path = self.img_file_path[:, index]
        data_info_ = self.data_info[self.data_info['exam_id'] == exam_id]  # data_info_should have four images
        patient_id = data_info_['patient_id'].iloc[0]
        PATH_R_CC = data_info_['PATH_R_CC'].iloc[0]
        PATH_R_MLO = data_info_['PATH_R_MLO'].iloc[0]
        PATH_L_CC = data_info_['PATH_L_CC'].iloc[0]
        PATH_L_MLO = data_info_['PATH_L_MLO'].iloc[0]
        # exam_date = data_info_['exam_date']
        # risk_label, right_risk_label, left_risk_label, follow_up_label
        risk_label = data_info_['years_to_cancer'].iloc[0]
        right_risk_label = data_info_['years_to_cancer_right'].iloc[0]
        left_risk_label = data_info_['years_to_cancer_left'].iloc[0]
        follow_up_label = data_info_['years_to_last_followup'].iloc[0]

        imgs = []
        times = []
        m_tumor_infos = []
        label_lib = ['risks', 'risks_r', 'risks_l',
                     'history', 'history_r', 'history_l',
                     'next_cancer_position', 'previous_cancer_position',
                     'next_cancer_type', 'next_cancer_type_left', 'next_cancer_type_right',
                     'previous_cancer_type', 'previous_cancer_type_left', 'previous_cancer_type_right',
                     'previous_cancer_location', 'previous_cancer_location_left', 'previous_cancer_location_right',
                     'next_cancer_location', 'next_cancer_location_left', 'next_cancer_location_right',
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
        labels = {}
        for label_ in label_lib:
            if label_ != 'age':
                labels[label_] = torch.from_numpy(np.asarray(-1, dtype='int64'))
            else:
                labels[label_] = torch.from_numpy(np.asarray(-1, dtype='float32'))

        # m_labels = {}
        # for label_ in label_lib:
        #     m_labels[label_] = []

        tumor_infos = torch.ones(2, 29) * -1

        num_exam = random.randint(1, self.num_time_points)

        for i in range(self.num_time_points):
            tumor_info = torch.ones(2, 29) * -1
            if i+1 <= num_exam:
                case = torch.ones(4, 1, *self.img_size) * -1
                # img_R_CC = self.__getimg(path[0], flip=False)
                img_R_CC = self.__getimg(PATH_R_CC, flip=False)
                if img_R_CC is not None:
                    case[0, :, :, :] = img_R_CC
                # img_R_MLO = self.__getimg(path[1], flip=False)
                img_R_MLO = self.__getimg(PATH_R_MLO, flip=False)
                if img_R_MLO is not None:
                    case[1, :, :, :] = img_R_MLO
                # img_L_CC = self.__getimg(path[2])
                img_L_CC = self.__getimg(PATH_L_CC)
                if img_L_CC is not None:
                    case[2, :, :, :] = img_L_CC
                # img_L_MLO = self.__getimg(path[3])
                img_L_MLO = self.__getimg(PATH_L_MLO)
                if img_L_MLO is not None:
                    case[3, :, :, :] = img_L_MLO
                imgs.append(case)
                times.append(0.0)
            else:
                imgs.append(torch.zeros(4, 1, *self.img_size))
                times.append(-1)
            m_tumor_infos.append(tumor_info)

        times = torch.as_tensor(times)
        imgs = torch.stack(imgs)
        m_tumor_infos = torch.stack(m_tumor_infos)
        return {'imgs': imgs,
                'times': times,
                'lens': 1,
                'risk': risk_label,
                'right_risk': right_risk_label,
                'left_risk': left_risk_label,
                'labels': labels,
                'follow_up': follow_up_label,
                'tumor_infos': tumor_infos,
                'm_tumor_infos': m_tumor_infos,
                'patient_id': patient_id,
                'exam_id': exam_id,
                }

    def __len__(self):
        return len(self.exam_ids)

    def __getimg(self, path, flip=False):
        # img_path = path
        img_path = '{}{}'.format(self.file_folder, path)
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if image is not None:
            image = cv2.resize(image, (512, 1024))
            if flip:
                image = cv2.flip(image, 1)
            image = imgunit8(image)
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
        else:
            logging.info('{} wrong!!!'.format(img_path))
            image = None

        return image
