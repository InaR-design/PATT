import os, argparse
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from dataset.ImbalanceImageNet import LT_Dataset
from models.resnet_imagenet import ResNet50
from dataset.imagenet_ood import ImageNet_ood
from sklearn.preprocessing import MinMaxScaler
from utils.utils import *
from utils.ltr_metrics import *

from test import get_measures
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
def get_energy_score(logits):
    scores = torch.logsumexp(logits, dim=1)
    return -scores

def get_msp_scores(logits):

    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values

    scores = -msp # The larger MSP, the smaller uncertainty

    return scores

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr_out = average_precision_score(labels, examples)
    labels_rev = np.zeros(len(examples), dtype=np.int32)
    labels_rev[len(pos):] += 1
    aupr_in = average_precision_score(labels_rev, -examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr_in, aupr_out, fpr, pos.mean(), neg.mean()

def val_imagenet():
    '''
    Evaluate ID acc and OOD detection on ImageNet
    '''
    finalScore = torch.zeros(2048).cuda()
    model.eval()
    if args.FC:
        with torch.no_grad():
            for batch_idx, ((in_data, labels), (ood_data, ood_labels)) in enumerate(zip(balanced_loader, ood_trainloader)):
                print(batch_idx)
                in_data, labels = in_data.cuda(), labels.cuda()
                ood_data, ood_labels = ood_data.cuda(), ood_labels.cuda()
                # forward:
                #all_labels = torch.cat([labels, ood_labels], dim=0)
                _, f_in = model(in_data)
                ood_logits, f_ood = model(ood_data)
                virtual_labels = ood_logits.max(1)[1]
                #calibration
                finalScore = model.Classbalanced_Calibration(f_in, f_ood, virtual_labels, labels, priors=prior, batchs = batch_idx, num_classes=num_classes, final_score = finalScore)
            
            finalScore = finalScore.cpu().numpy().reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0,2))
            finalScore = scaler.fit_transform(finalScore).ravel()
            finalScore = torch.from_numpy(finalScore).cuda()
            print('Calibration finished')
    # test:
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    prob_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, p4 = model(images)
            if args.FC:
                logits = model.forward_classifier(p4 * finalScore)

            probs = F.softmax(logits, dim=1)
            scores = get_energy_score(logits)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
            prob_list.append(probs.detach().cpu().numpy())
    

    test_acc = test_acc_meter.avg
    assert len(score_list) == len(labels_list)
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    
    
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

    # confidence distribution of correct samples:
    ood_score_list, sc_labels_list, out_prob_list = [], [], []

    with torch.no_grad():
        for images, sc_labels in ood_loader:
            images, sc_labels = images.cuda(), sc_labels.cuda()
            logits, p4 = model(images)
            if args.FC:
                logits = model.forward_classifier(p4 * finalScore)

            probs = F.softmax(logits, dim=1)
            scores = get_energy_score(logits)
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
            sc_labels_list.append(sc_labels.detach().cpu().numpy())
            out_prob_list.append(probs.detach().cpu().numpy())
    
    ood_scores = np.concatenate(ood_score_list, axis=0)
    sc_labels = np.concatenate(sc_labels_list, axis=0)
    print('in_scores:', in_scores.shape)
    print('ood_scores:', ood_scores.shape)

    # print:
    auroc, aupr_in, aupr_out, fpr95, id_meansocre, ood_meanscore = get_measures(ood_scores,in_scores)
    ood_detectoin_str = 'auroc: %.4f, aupr_in: %.4f, aupr_out: %.4f, fpr95: %.4f, ood_meanscore: %.4f, id_meansocre: %.4f' % (auroc, aupr_in, aupr_out, fpr95, ood_meanscore, id_meansocre)
    print(ood_detectoin_str)
    fp.write(ood_detectoin_str + '\n')
    fp.flush()
    fp.close()

    classwise_results_dir = os.path.join(save_dir, 'classwise_results')
    create_dir(classwise_results_dir)
    # classwise acc:
    acc_each_class = np.full(num_classes, np.nan)
    for i in range(num_classes):
        _pred = in_preds[in_labels==i]
        _label = in_labels[in_labels==i]
        _N = np.sum(in_labels==i)
        acc_each_class[i] = np.sum(_pred==_label) / _N
    np.save(os.path.join(classwise_results_dir, 'ACC_each_class.npy'), acc_each_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test an ImageNet Classifier')
    parser.add_argument('--seed', default=3407, type=int, help='fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--gpu', default='0', help='which GPU to use.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of threads for data loader')
    # dataset:
    parser.add_argument('--dataset', '--ds', default='imagenet', choices=['imagenet'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='./datasets', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='imagenet-10k', choices=['imagenet-10k'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet50', choices=['ResNet50'], help='which model to use')
    # 
    parser.add_argument('--test_batch_size', '--tb', type=int, default=100)
    parser.add_argument('--ckpt_path', default='/home/imt-3090-2/inar/COCL/result_copy/imagenet-lt/ResNet50/e100-b32-64-sgd-lr0.1-wd5e-05_Lambda1 0.02-Lambda2 0-Lambda5 0.5/replay2')
    parser.add_argument('--tnorm', default=False, help='If true, use t-norm for LT inference')
    parser.add_argument('--num_cal', default=30000, type=int, help='num for calibration.')
    parser.add_argument('--FC', default=False, help='If true, use outlier-class-aware logit calibration for LT inference')
    parser.add_argument('--txt_train', '--txtt', default='./dataset/ImageNet_LT_train.txt', help='txt path for train')
    parser.add_argument('--txt_val', '--txtv', default='./dataset/ImageNet_LT_val.txt', help='txt path for val')
    parser.add_argument('--iidp', '--iidp', default='/mnt/data/melon/imagenet/imagenet12', help='path for imagenet1k')
    parser.add_argument('--iodp', '--iodp', default='/mnt/data/IMAGENET', help='path for imagenet10k')
    args = parser.parse_args()
    print(args)

    # ============================================================================
    # fix random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.FC:
        save_dir = os.path.join(args.ckpt_path, 'FC', args.dout)
    elif args.tnorm:
        save_dir = os.path.join(args.ckpt_path, 'tnorm', args.dout)
    else:
        save_dir = os.path.join(args.ckpt_path, 'normal', args.dout)
    create_dir(save_dir)

    # data prepossessing:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    num_classes = 1000
    train_set = LT_Dataset(
            root=args.iidp,
            txt=args.txt_train,
            transform=train_transform, train=False)
    test_set = LT_Dataset(
            root=args.iidp,
            txt=args.txt_val,
            transform=test_transform, train=False)
    balance_set = Subset(LT_Dataset(
            root=args.iidp,
            txt=args.txt_train,
            transform=test_transform, train=False,class_balance=True), list(range(args.num_cal)))
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    balanced_loader = DataLoader(balance_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    din_str = 'Din is %s with %d images' % (args.dataset, len(test_set))
    print(din_str)
    
    ood_trainset = Subset(ImageNet_ood(root = args.iodp, transform=train_transform, txt="./dataset/imagenet_extra_1k_wnid_list_picture.txt"), list(range(args.num_cal)))
    ood_set = ImageNet_ood(root = args.iodp, transform=train_transform, txt="./dataset/imagenet_ood_test_1k_wnid_list_picture.txt")
    ood_trainloader = DataLoader(ood_trainset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False, pin_memory=True)
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    dout_str = 'Dout is %s with %d images' % (args.dout, len(ood_set))
    print(dout_str)

    img_num_per_cls = np.array(train_set.cls_num_list)
    prior = img_num_per_cls / np.sum(img_num_per_cls)
    prior = torch.from_numpy(prior).float().cuda()

    # model:
    model = ResNet50(num_classes=num_classes, return_features=False).cuda()

    # load model:
    ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'))['model']
    model.load_state_dict(ckpt)   
    model.requires_grad_(False)

    # log file:
    if args.tnorm:
        '''
        Decoupling representation and classifier for long-tailed recognition. ICLR, 2020.
        '''
        w = model.fc.weight.data
        w_row_norm = torch.norm(w, p='fro', dim=1)
        print(w_row_norm)
        model.fc.weight.data = w / (2*w_row_norm[:,None])
        model.fc.bias.zero_()
    test_result_file_name = 'test_results.txt'
    fp = open(os.path.join(save_dir, test_result_file_name), 'a+')
    fp.write('\n===%s===\n' % (args.dout))
    fp.write(din_str + '\n')
    fp.write(dout_str + '\n')


    val_imagenet()