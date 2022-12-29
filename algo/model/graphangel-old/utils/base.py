import numpy as np
import torch as th
from sklearn.metrics import accuracy_score, roc_auc_score


def glorot(shape, scale=1.0):
    init_range = np.sqrt(6.0 / (shape[-1] + shape[-2])) * scale
    initial = np.random.uniform(-init_range, init_range, shape)
    return th.Tensor(initial)


def cal_auc(pred, label, num):
    # pred, label: batch_size, label_num
    # split result according to label_num
    _auc = []
    for _num in range(num):
        if np.sum(label[:, _num]) == 0:
            padding = [1 for _ in range(num)]
            pred = np.vstack([pred, padding])
            label = np.vstack([label, padding])
        elif np.sum(label[:, _num]) == len(label):
            padding = [0 for _ in range(num)]
            pred = np.vstack([pred, padding])
            label = np.vstack([label, padding])
        _auc.append(roc_auc_score(y_score=np.array(pred[:, _num]), y_true=np.array(label[:, _num])))
    return np.mean(_auc)


def cal_acc(pred, label, num):
    # pred, label: batch_size, label_num
    _acc = []
    for _num in range(num):
        for _index, _pred in enumerate(pred):
            if _pred[_num] >= 0.5:
                pred[_index, _num] = 1
            else:
                pred[_index, _num] = 0
        _acc.append(accuracy_score(y_pred=np.array(pred[:, _num]), y_true=np.array(label[:, _num])))
    return np.mean(_acc)
