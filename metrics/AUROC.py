import torch
import warnings
import numpy as np 
from sklearn.metrics import roc_auc_score

def AUC_metric(pred, target, num_classes):
    
    mask = torch.stack(
        [target.eq(i) for i in range(num_classes)],
        1).squeeze()  # mask
    N = mask.sum(0)  # [类1的样本数目, ...]

    exist_class_num = torch.sum(N != 0)

    metric = 0
    loss = torch.Tensor([0.]).cuda()

    for i in range(num_classes):
        if N[i] == 0:
            continue
        mask_i = mask[:, i]

        f_i = pred[mask_i, :][:, i]
        f_not_i = pred[~mask_i, :][:, i]

        i_sample_num = f_i.shape[0]
        not_i_sample_num = f_not_i.shape[0]

        matrix_size = torch.Size((not_i_sample_num, i_sample_num))

        f_i = f_i.unsqueeze(0)
        f_not_i = f_not_i.unsqueeze(1)

        diff = (f_i - f_not_i)
        N_j = N[target[~mask_i]].unsqueeze(1).expand(matrix_size)
        metric += torch.sum( ((diff > 0).float() + (diff == 0).float()/2) / N_j / N[i]) / (exist_class_num - 1)
    
    metric /= exist_class_num
    return metric
    


def AUC(y_true, y_pred, multi_type='ovo', acc=True):
    """
        Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
        Note:
            This function can be only used with binary, multiclass AUC (either 'ova' or 'ovo').

    """
    if not isinstance(y_true, np.ndarray):
        warnings.warn("The type of y_ture must be np.ndarray")
        y_true = np.asarray(y_true)
    
    if not isinstance(y_pred, np.ndarray):
        warnings.warn("The type of y_pred must be np.ndarray")
        y_pred = np.asarray(y_pred)
    
    if len(np.unique(y_true)) == 2:
        assert len(y_pred) == len(y_true)
        return roc_auc_score(y_true=y_true, y_pred=y_pred)
    elif multi_type == 'ova':
        return roc_auc_score(y_true=y_true, y_pred=y_pred, multi_class='ovr')
    elif multi_type == 'ovo':
        if acc:
            return fast_multi_class_auc_score(y_true=y_true, y_pred=y_pred)
        else:
            return multi_class_auc_score(y_true=y_true, y_pred=y_pred)
    else:
        raise NotImplementedError('multi_type must be contained in [ova, ovo]')

def multi_class_auc_score(y_true, y_pred, **kwargs):
    n = y_true.shape[1]

    def bin_auc(label, pred, i, j):
        msk1 = label[:, i] == 1
        msk2 = label[:, j] == 1
        y1 = pred[msk1, i]
        y2 = pred[msk2, i]
        return np.mean([ix > jx for ix in y1 for jx in y2])

    return np.mean([
        bin_auc(y_true, y_pred, i, j) for i in range(n) for j in range(n)
        if i != j
    ])

def fast_multi_class_auc_score(y_true, y_pred, **kwargs):
    y_true = np.argmax(y_true, 1)
    classes = np.unique(y_true)
    num_class = len(classes)
    sum_cls = np.array([(y_true == i).sum() for i in range(num_class)])

    def bin_auc(label, pred, idx, sum_cls):
        pd = pred[:, idx]
        lbl = label
        r = np.argsort(pd)
        lbl = lbl[r]
        sum_cls = sum_cls[lbl]

        loc_idx = np.where(lbl == idx)[0][::-1]
        weight = np.zeros((len(lbl)))
        for i in range(len(loc_idx) - 1):
            weight[loc_idx[i+1] + 1:loc_idx[i]] = i + 1
        weight[:loc_idx[-1]] = len(loc_idx)

        res = (weight / sum_cls).sum() / len(loc_idx) / (num_class - 1)

        return res

    return np.mean([
        bin_auc(y_true, y_pred, idx, sum_cls) for idx in range(num_class)
    ])