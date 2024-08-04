'''
Adapted from https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/utils.py
which uses GPL-3.0 license.
'''
import torch
import numpy as np 


def shot_acc(preds, labels, train_class_count, acc_per_cls=False):

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    num_classes = len(train_class_count)

    test_class_count = [np.nan] * num_classes
    class_correct = [np.nan] * num_classes
    for l in range(num_classes):
        test_class_count[l] = len(labels[labels == l])
        class_correct[l] = (preds[labels == l] == labels[labels == l]).sum()

    if num_classes <= 100: # e.g. On CIFAR10/100
        many_shot_thr = train_class_count[int(0.34*num_classes)]
        low_shot_thr = train_class_count[int(0.67*num_classes)]
    else:
        many_shot_thr=100
        low_shot_thr=20
    # print(many_shot_thr, low_shot_thr)

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(num_classes):
        if test_class_count[i] == 0:
            assert class_correct[i] == 0
            _acc_class_i = np.nan
        else:
            _acc_class_i = class_correct[i] / test_class_count[i]
        if train_class_count[i] > many_shot_thr:
            many_shot.append(_acc_class_i)
        elif train_class_count[i] < low_shot_thr:
            low_shot.append(_acc_class_i)
        else:
            median_shot.append(_acc_class_i)    

    # print('many_shot:', many_shot)
    # print('median_shot:', median_shot)
    # print('low_shot:', low_shot)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot), class_accs
    else:
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot)
    