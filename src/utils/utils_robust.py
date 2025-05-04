import math
import torch
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from lifelines import KaplanMeierFitter
import seaborn as sns
from .c_index import concordance_index

def imgunit8(img):
    # mammogram_dicom = img
    # orig_min = mammogram_dicom.min()
    # orig_max = mammogram_dicom.max()
    # target_min = 0.0
    # target_max = 255.0
    # mammogram_scaled = (mammogram_dicom-orig_min)*((target_max-target_min)/(orig_max-orig_min))+target_min
    mammogram_scaled = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint8)
    # print(mammogram_uint8_by_function.max(), mammogram_uint8_by_function.min())
    # mammogram_uint8_by_function = mammogram_uint8_by_function
    return mammogram_uint8_by_function


def imgunit16(img):
    mammogram_dicom = img
    orig_min = mammogram_dicom.min()
    orig_max = mammogram_dicom.max()
    target_min = 0.0
    target_max = 65535.0
    mammogram_scaled = (mammogram_dicom-orig_min)*((target_max-
    target_min)/(orig_max-orig_min))+target_min
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint16)
    return mammogram_uint8_by_function

def split_data_id(data_info, split_ratio):
    split_ratio = split_ratio
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data_info)
    split = int(np.floor(split_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(data_info)
    indices_1, indices_2 = data_info[:split], data_info[split:]

    return indices_1, indices_2


def comput_cm(all_labels, all_predicted):
    # confusion_matrix
    y_true = all_labels
    y_true = np.array(y_true)
    y_true = y_true.reshape(-1, 1)
    y_pred = all_predicted
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(-1, 1)
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm

def get_y_ture_pred(all_labels, all_predicted):
    y_true = all_labels
    y_true = np.array(y_true)
    y_true = y_true.reshape(-1, 1)
    y_pred = all_predicted
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(-1, 1)
    return y_true, y_pred


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

def prob_to_score(prob, max_followup=5):
    # print('prob')
    # for i in range(15):
    score = np.zeros_like(prob)[:, 0:max_followup]
    for i in range(max_followup):
        # i_ = -(i + 1)
        # score[:, i] = prob[:, i_]
        for i_in in range(i+1):
            i_ = -(i_in + 1)
            score[:, i] += prob[:, i_]
    return score

def get_censor_info(labels, followups, num_classes=6 ,max_followup=5):
    labels = np.squeeze(labels)
    followups = np.squeeze(followups)
    labels_ = num_classes - 1 - labels
    years_to_cancer = labels_
    years_to_last_followup = followups

    any_cancer = years_to_cancer < max_followup
    # cancer_key = "years_to_cancer"

    y = any_cancer
    shape = np.shape(any_cancer)[0]
    y_seq = np.zeros([shape, max_followup])

    time_at_event = np.zeros_like(years_to_cancer)
    y_mask = np.zeros_like(y_seq)

    for i in range(shape):
        if y[i]:
            time_at_event[i] = int(years_to_cancer[i])
            y_seq[i, time_at_event[i]:] = 1
        else:
            time_at_event[i] = int(min(years_to_last_followup[i], max_followup) - 1)

        # y_mask[i, :] = np.array([1] * (time_at_event+1) + [0]* (max_followup - (time_at_event+1)))
        y_mask[i, :] = np.array([1] * (time_at_event[i] +1) + [0]* (max_followup - (time_at_event[i] +1)))
    # y_mask = np.array([1] * (time_at_event+1) + [0]* (max_followup - (time_at_event+1)))
    # assert len(y_mask) == max_followup
    return any_cancer, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event


def get_censoring_dist(times, event_observed):
    # _dataset = train_dataset.dataset
    # times, event_observed = [d['time_at_event'] for d in _dataset], [d['y'] for d in _dataset]
    # times, event_observed = [d for d in times], [d for d in event_observed]
    times = list(times)
    all_observed_times = set(list(times))
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    return censoring_dist


def compute_auc_metrics_given_curve(probs, censor_times, golds, max_followup, censor_distribution):
    metrics = {}
    sample_sizes = {}
    for followup in range(max_followup):
        min_followup_if_neg = followup + 1

        auc, golds_for_eval = compute_auc_x_year_auc(probs, censor_times, golds, followup)
        key = min_followup_if_neg
        metrics[key] = auc
        sample_sizes[key] = golds_for_eval
    try:
        c_index = concordance_index(censor_times, probs, golds, censor_distribution)
    except Exception as e:
            warnings.warn("Failed to calculate C-index because {}".format(e))
            c_index = 'NA'

    metrics['c_index'] = c_index
    end_probs = np.array(probs)[:,-1].tolist()
    sorted_golds = [g for p,g in sorted( zip(end_probs, golds))]
    metrics['decile_recall'] = sum( sorted_golds[-len(sorted_golds)//10:]) / sum(sorted_golds)
    return metrics, sample_sizes


def compute_auc_x_year_auc(probs, censor_times, golds, followup):

    def include_exam_and_determine_label( prob_arr, censor_time, gold):
        valid_pos = gold and censor_time <= followup
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(prob_arr, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(label)

    try:
        auc = metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate AUC because {}".format(e))
        auc = 'NA'

    return auc, golds_for_eval


def comput_yala_metrics(args, all_labels_np, all_probabilities_np, all_followups_np):
    max_followup = args.test_num_classes - 1
    # all_followups_np = np.ones_like(all_labels_np) * (max_followup + 1)
    score = prob_to_score(all_probabilities_np, max_followup=max_followup)
    y, y_seq, y_mask, time_at_event = get_censor_info(all_labels_np, all_followups_np, num_classes=args.num_classes,
                                                      max_followup=max_followup)
    censor_distribution = get_censoring_dist(time_at_event, y)

    probs = score
    censor_times = time_at_event
    golds = y
    censor_distribution = censor_distribution

    metrics, sample_sizes = compute_auc_metrics_given_curve(probs, censor_times, golds, max_followup,
                                                            censor_distribution)
    return metrics


def polt_roc_curve(args, all_labels, all_probabilities, all_followups, poltroc=False, name='best model'):
    # plot ROC

    """
    clean_data_without_enough_followup: clean data without enough followup
    """
    clean_data_without_enough_followup = args.clean_data_without_enough_followup


    if 'Test' in str(name):
        num_classes = args.test_num_classes
    else:
        num_classes = args.num_classes

    # num_classes = args.test_num_classes

    all_labels_np = np.array(all_labels)
    all_probabilities_np = np.array(all_probabilities)
    all_followups_np = np.array(all_followups)
    all_labels_np = all_labels_np.reshape(-1, 1)
    all_probabilities_np = all_probabilities_np.reshape(-1, args.num_classes)
    # all_probabilities_np = all_probabilities_np[:, 1]
    all_followups_np = all_followups_np.reshape(-1, 1) + 1

    # # -------- use yala's code to compute c-index and auc for checking -------- # #
    metrics = comput_yala_metrics(args, all_labels_np, all_probabilities_np, all_followups_np)
    # # -------- ------------------------------------------------------- -------- # #

    roc_auc_overall = []
    for n in range(num_classes - 1):
        # print('followups check is ', num_classes - n - 1)

        idx_with_label, _ = np.where(all_followups_np >= (num_classes - n - 1))

        if clean_data_without_enough_followup:
            all_labels_np_clean = all_labels_np[idx_with_label]
            all_probabilities_np_clean = all_probabilities_np[idx_with_label, :]
        else:
            all_labels_np_clean = all_labels_np
            all_probabilities_np_clean = all_probabilities_np

        all_probabilities_np_ = np.zeros_like(all_labels_np_clean)
        all_labels_np_ = all_labels_np_clean.copy()
        for classes in range(args.num_classes - num_classes + n + 1, args.num_classes):
            # all_probabilities_np__ = all_probabilities_np[:, classes]
            all_probabilities_np_ = all_probabilities_np_ + np.expand_dims(all_probabilities_np_clean[:, classes], 1)

        # all_labels_np_ = all_labels_np_ - (args.num_classes-num_classes+n+10)
        all_labels_np_[all_labels_np_ < (args.num_classes - num_classes + n + 1)] = 0
        all_labels_np_[all_labels_np_ != 0] = 1
        fpr, tpr, thresholds = roc_curve(all_labels_np_, all_probabilities_np_)
        roc_auc = auc(fpr, tpr)
        roc_auc_overall.append(roc_auc)
        # if poltogether:

        if poltroc:
            # plt.figure(dpi=300)
            # plt.plot(fpr, tpr, label='{}--AUC={:.4f}'.format(name, roc_auc), lw=2, alpha=.8)
            plt.plot(fpr, tpr, label='{:.0f} Y-AUC {:.2f}'.format(num_classes - n - 1, roc_auc), lw=2,
                     alpha=.8)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
            plt.title('BC risk ROC Curve')
            plt.legend(loc='lower right')
            # plt.show()
            plt.savefig(args.results_dir_fold + '/ROC_{}.png'.format(name), bbox_inches='tight',
                        # transparent=True,
                        dpi=300)
    plt.close()

    c_index = metrics['c_index']
    decile_recall = metrics['decile_recall']

    return roc_auc_overall, c_index, decile_recall


def compute_mean_ci(interp_tpr, confidence=0.95):

    tpr_mean = np.zeros((interp_tpr.shape[1]), dtype='float32')
    tpr_lb = np.zeros((interp_tpr.shape[1]), dtype='float32')
    tpr_up = np.zeros((interp_tpr.shape[1]), dtype='float32')
    tpr_std = np.zeros((interp_tpr.shape[1]), dtype='float32')

    Pz = (1.0 - confidence) / 2.0

    for i in range(interp_tpr.shape[1]):
        # get sorted vector
        vec = interp_tpr[:, i]
        vec.sort()

        tpr_mean[i] = np.average(vec)
        tpr_lb[i] = vec[int(math.floor(Pz * len(vec)))]
        tpr_up[i] = vec[int(math.floor((1.0 - Pz) * len(vec)))]
        tpr_std[i] = np.std(vec)

    return tpr_mean, tpr_lb, tpr_up, tpr_std


def compute_mean_ci_(c_index_bs):
    c_index_bs = np.array(c_index_bs)

    # get mean
    c_index_bs_mean = np.mean(c_index_bs)
    # get median
    c_index_bs_median = np.percentile(c_index_bs, 50)
    # get 95% interval
    alpha = 100 - 95
    c_index_bs_lower_ci = np.percentile(c_index_bs, alpha / 2)
    c_index_bs_upper_ci = np.percentile(c_index_bs, 100 - alpha / 2)
    return c_index_bs_mean, c_index_bs_lower_ci, c_index_bs_upper_ci, c_index_bs_median


def polt_roc_curve_with_CI(args, all_labels, all_probabilities, all_followups, poltroc=False, name='best model'):
    # plot ROC
    # if 'Test' in str(name):
    #     num_classes = args.test_num_classes
    # else:
    #     num_classes = args.num_classes
    """
    clean_data_without_enough_followup: clean data without enough followup
    """
    clean_data_without_enough_followup = args.clean_data_without_enough_followup

    num_classes = args.test_num_classes
    all_labels_np = np.array(all_labels)
    all_probabilities_np = np.array(all_probabilities)
    all_followups_np = np.array(all_followups)
    all_labels_np = all_labels_np.reshape(-1, 1)
    all_probabilities_np = all_probabilities_np.reshape(-1, args.num_classes)
    all_followups_np = all_followups_np.reshape(-1, 1) + 1

    n_iterations = 1000
    roc_auc_overall_bs = []
    fpr_overall_bs = []
    tpr_overall_bs = []
    c_index_bs = []
    decile_recall_bs = []
    for i in range(n_iterations):
        all_probabilities_np_bs, all_labels_np_bs, all_followups_np_bs = resample(
            all_probabilities_np, all_labels_np, all_followups_np, replace=True, random_state=i
        )
        metrics_ = comput_yala_metrics(args, all_labels_np_bs, all_probabilities_np_bs, all_followups_np_bs)
        c_index = metrics_['c_index']
        decile_recall = metrics_['decile_recall']
        c_index_bs.append(c_index)
        decile_recall_bs.append(decile_recall)
        roc_auc_overall_ = []
        fpr_overall_ = []
        tpr_overall_ = []
        for n in range(num_classes - 1):
            if clean_data_without_enough_followup:
                idx_with_label, _ = np.where(all_followups_np_bs >= (num_classes - n - 1))
                all_labels_np_bs_clean = all_labels_np_bs[idx_with_label]
                all_probabilities_np_bs_clean = all_probabilities_np_bs[idx_with_label, :]
            else:
                all_labels_np_bs_clean = all_labels_np_bs
                all_probabilities_np_bs_clean = all_probabilities_np_bs

            all_probabilities_np_bs_ = np.zeros_like(all_labels_np_bs_clean)
            all_labels_np_bs_ = all_labels_np_bs_clean.copy()
            for classes in range(args.num_classes - num_classes + n + 1, args.num_classes):
                # all_probabilities_np__ = all_probabilities_np[:, classes]
                all_probabilities_np_bs_ = all_probabilities_np_bs_ + np.expand_dims(
                    all_probabilities_np_bs_clean[:, classes], 1)
            all_labels_np_bs_[all_labels_np_bs_ < (args.num_classes - num_classes + n + 1)] = 0
            all_labels_np_bs_[all_labels_np_bs_ != 0] = 1
            fpr, tpr, thresholds = roc_curve(all_labels_np_bs_, all_probabilities_np_bs_)
            fpr_overall_.append(fpr)
            tpr_overall_.append(tpr)
            roc_auc = auc(fpr, tpr)
            roc_auc_overall_.append(roc_auc)
            fpr_overall_bs.append(fpr_overall_)
            tpr_overall_bs.append(tpr_overall_)
        roc_auc_overall_bs.append(roc_auc_overall_)

    # compute statistic
    all_fpr = np.linspace(0, 1, num=10000)
    all_auc_cls_ci = []
    tpr_overall_lb = []
    tpr_overall_up = []
    tpr_overall_mean = []
    for i in range(num_classes - 1):
        n_all_auc_cls_ = []
        n_tpr_overall_ = []
        for n in range(n_iterations):
            n_all_auc_cls_.append(roc_auc_overall_bs[n][i])
            n_tpr_overall_.append(np.interp(all_fpr, fpr_overall_bs[n][i], tpr_overall_bs[n][i]))
        n_tpr_overall_ = np.array(n_tpr_overall_)
        # get mean
        n_all_auc_cls_mean = np.mean(n_all_auc_cls_)
        tpr_overall_mean.append(np.mean(n_tpr_overall_, axis=0))
        # get median
        n_all_auc_cls_median = np.percentile(n_all_auc_cls_, 50)
        # get 95% interval
        alpha = 100 - 95
        n_all_auc_cls_lower_ci = np.percentile(n_all_auc_cls_, alpha / 2)
        n_all_auc_cls_upper_ci = np.percentile(n_all_auc_cls_, 100 - alpha / 2)
        all_auc_cls_ci.append([n_all_auc_cls_mean, n_all_auc_cls_lower_ci, n_all_auc_cls_upper_ci, n_all_auc_cls_median])
        tpr_overall_lb.append(np.percentile(n_tpr_overall_, alpha / 2, axis=0))
        tpr_overall_up.append(np.percentile(n_tpr_overall_, 100 - alpha / 2, axis=0))

    palette = plt.get_cmap('tab20')
    roc_auc_overall = []

    # # -------- use yala's code to compute c-index and auc for checking -------- # #
    metrics = comput_yala_metrics(args, all_labels_np, all_probabilities_np, all_followups_np)
    # # -------- ------------------------------------------------------- -------- # #

    for n in range(num_classes - 1):
        if clean_data_without_enough_followup:
            idx_with_label, _ = np.where(all_followups_np >= (num_classes - n - 1))
            all_labels_np_clean = all_labels_np[idx_with_label]
            all_probabilities_np_clean = all_probabilities_np[idx_with_label, :]
        else:
            all_labels_np_clean = all_labels_np
            all_probabilities_np_clean = all_probabilities_np

        all_probabilities_np_ = np.zeros_like(all_labels_np_clean)
        all_labels_np_ = all_labels_np_clean.copy()
        for classes in range(args.num_classes - num_classes + n + 1, args.num_classes):
            # all_probabilities_np__ = all_probabilities_np[:, classes]
            all_probabilities_np_ = all_probabilities_np_ + np.expand_dims(all_probabilities_np_clean[:, classes], 1)

        # all_labels_np_ = all_labels_np_ - (args.num_classes-num_classes+n+10)
        all_labels_np_[all_labels_np_ < (args.num_classes - num_classes + n + 1)] = 0
        all_labels_np_[all_labels_np_ != 0] = 1
        fpr, tpr, thresholds = roc_curve(all_labels_np_, all_probabilities_np_)
        roc_auc = auc(fpr, tpr)
        roc_auc_overall.append(
            [roc_auc, all_auc_cls_ci[n][0], all_auc_cls_ci[n][1], all_auc_cls_ci[n][2], all_auc_cls_ci[n][3]])
        # if poltogether:

        if poltroc:
            clr = palette(n)
            # plt.figure(dpi=300)
            ax = plt.gca()
            # plt.plot(fpr, tpr, label='{}--AUC={:.4f}'.format(name, roc_auc), lw=2, alpha=.8)
            # plt.plot(fpr, tpr, label='BC risk within {:.0f} Y--AUC={:.4f} [{:.4f}, {:.4f}]'.format(
            #     num_classes - n - 1, roc_auc, all_auc_cls_ci[n][1], all_auc_cls_ci[n][2]), color=clr, lw=1, alpha=.8)
            plt.plot(fpr, tpr, label='{:.0f} Y-AUC: {:.2f} [{:.2f},{:.2f}]'.format(
                num_classes - n - 1, all_auc_cls_ci[n][0], all_auc_cls_ci[n][1], all_auc_cls_ci[n][2]),
                     color=clr, lw=1.5, alpha=.8)
            plt.plot(all_fpr, tpr_overall_mean[n], color=clr, ls='--', lw=0.5, alpha=.5)
            plt.plot(all_fpr, tpr_overall_lb[n], color=clr, ls='-.', lw=0.5, alpha=.5)
            plt.plot(all_fpr, tpr_overall_up[n], color=clr, ls='-.', lw=0.5, alpha=.5)
            ax.fill_between(all_fpr, tpr_overall_lb[n], tpr_overall_up[n], facecolor=clr, alpha=0.2)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
            plt.title('BC risk ROC Curve')
            plt.legend(loc='lower right')
            # plt.show()
            plt.savefig(args.results_dir_fold + '/ROC_{}.png'.format(name), bbox_inches='tight',
                        # transparent=True,
                        dpi=300)
    plt.close()

    c_index = metrics['c_index']
    c_index_bs_mean, c_index_bs_lower_ci, c_index_bs_upper_ci, c_index_bs_median = compute_mean_ci_(c_index_bs)
    c_index_overall = [c_index, c_index_bs_mean, c_index_bs_lower_ci, c_index_bs_upper_ci, c_index_bs_median]
    decile_recall = metrics['decile_recall']
    decile_recall_bs_mean, decile_recall_bs_lower_ci, decile_recall_bs_upper_ci, decile_recall_bs_median = compute_mean_ci_(
        decile_recall_bs)
    decile_recall_overall = [decile_recall, decile_recall_bs_mean, decile_recall_bs_lower_ci, decile_recall_bs_upper_ci,
                             decile_recall_bs_median]

    return roc_auc_overall, c_index_overall, decile_recall_overall


def write_report_to_text(results_dir_fold, report, name):
    report_path = str(results_dir_fold + name)
    text_file = open(report_path, "w")
    text_file.write(report)
    text_file.close()


def save_checkpoint(results_dir_fold, state, is_best):

    filename = results_dir_fold + '/model_last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, results_dir_fold + '/model_best.pth.tar')
