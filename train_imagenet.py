import argparse, os, datetime, time
from sklearn.metrics import f1_score
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from dataset.cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from dataset.ImbalanceImageNet import LT_Dataset
from dataset.imagenet_ood import ImageNet_ood
from models.resnet import ResNet18, ResNet34
from models.resnet_imagenet import ResNet50

from utils.utils import *
from utils.ltr_metrics import shot_acc
from loss import *
from utils.randaugment import rand_augment_transform
from loss.logitadjust import LogitAdjust
from loss.proco import ProCoLoss

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--seed', default=3407, type=int, help='fix the random seed for reproduction. Default is 3407.')
    parser.add_argument('--replay', default='replay', help='repetitions for reproduction.')
    parser.add_argument('--gpu', default='0', help='which GPU to use.')
    parser.add_argument('--num_workers', '--cpus', default=4, help='number of threads for data loader')
    parser.add_argument('--dataset', '--ds', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--model', '--md', default='ResNet50', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='input batch size for training')
    parser.add_argument('--ood_batch_size', '--ob', type=int, default=64, help='OOD batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--Lambda1', default=0.02, type=float, help='OE loss term tradeoff hyper-parameter: 0.1 for CIFAR and 0.02 for ImageNet')
    parser.add_argument('--Lambda2', default=0.5, type=float, help='Temperature Scaling-based Logits Adjustment loss term tradeoff hyper-parameter:0.5')
    parser.add_argument('--num_ood_samples', default=300000, type=int, help='Number of OOD samples to use.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Tempearture in implicit semantic augmentation contrastive loss')
    parser.add_argument('--tem_scale', type=float, default=0.7, help='Temperature in temperature scaling-based logits adjustment loss')
    parser.add_argument('--headrate', default=0.4, type=float, help='percentage of head to use')
    parser.add_argument('--tailrate', default=0.4, type=float, help='percentage of tail to use')
    parser.add_argument('--save_root_path', '--srp', default='./results', help='data root path')
    parser.add_argument('--txt_train', '--txtt', default='./dataset/ImageNet_LT_train.txt', help='txt path for train')
    parser.add_argument('--txt_val', '--txtv', default='./dataset/ImageNet_LT_val.txt', help='txt path for val')
    parser.add_argument('--iidp', default='', help='path for imagenet1k')
    parser.add_argument('--iodp', default='', help='path for imagenet10k')
    parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
    parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
    parser.add_argument('--decay_epochs', '--de', default=[60,80], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    args = parser.parse_args()

    return args

def create_save_path():
    # mkdirs:
    opt_str = 'e%d-b%d-%d-%s-lr%s-wd%s' % (args.epochs, args.batch_size, args.ood_batch_size, args.opt, args.lr, args.wd)
    loss_str = 'Lambda1 %s-Lambda2 %s' % (args.Lambda1, args.Lambda2)
    exp_str = '%s_%s' % (opt_str, loss_str)
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s-lt' % (args.dataset)
    save_dir = os.path.join(args.save_root_path, dataset_str, args.model, exp_str, args.replay)
    create_dir(save_dir)
    print('Saving to %s' % save_dir)

    return save_dir, dataset_str


def train(args): 
    # get batch size:
    train_batch_size = args.batch_size 
    ood_batch_size = args.ood_batch_size 
    num_workers = args.num_workers

    save_dir = args.save_dir 
    device = 'cuda'

    # data preprocessing:
    rgb_mean = (0.485, 0.456, 0.406)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    ood_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                        transforms.Compose(augmentation_sim), ]
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    num_classes = 1000
    train_set = LT_Dataset(
        root=args.iidp,
        txt=args.txt_train,
        transform=train_transform, train=True)
    test_set = LT_Dataset(
        root=args.iidp,
        txt=args.txt_val,
        transform=val_transform, train=False)
        
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    
    
    ood_set = ImageNet_ood(root = args.iodp, transform=ood_train_transform, txt="./dataset/imagenet_extra_1k_wnid_list_picture.txt")
    ood_loader = DataLoader(ood_set, batch_size=ood_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
    
    # get prior distributions:
    img_num_per_cls = np.array(train_set.cls_num_list)
    prior = img_num_per_cls / np.sum(img_num_per_cls)
    prior = torch.from_numpy(prior).float().to(device)
    
    assert np.sum(img_num_per_cls) == len(train_set), 'Sum of image numbers per class %d neq total image number %d' % (np.sum(img_num_per_cls), len(train_set))
    
    # plot a curve showing the number of samples per class
    plt.plot(np.sort(img_num_per_cls)[::-1])
    plt.savefig(os.path.join(save_dir, 'img_num_per_cls.png'))
    plt.close()

    img_num_per_cls = torch.from_numpy(img_num_per_cls).float().to(device)

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes, return_features=True).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes, return_features=True).to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes, return_features=True).to(device)
    else:
        raise ValueError("illegal training model")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    criterion_ce = LogitAdjust(img_num_per_cls).cuda(args.gpu)
    criterion_scl = ProCoLoss(contrast_dim=1024, temperature=args.temperature, num_classes=num_classes).cuda(args.gpu)

    #resume
    if args.resume:
        ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])  
        start_epoch = ckpt['epoch']+1 
        training_losses = ckpt['training_losses']
        test_clean_losses = ckpt['test_clean_losses']
        f1s = ckpt['f1s']
        overall_accs = ckpt['overall_accs']
        many_accs = ckpt['many_accs']
        median_accs = ckpt['median_accs']
        low_accs = ckpt['low_accs']
    else:
        training_losses, test_clean_losses = [], []
        f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
        start_epoch = 0
    
    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        training_loss_meter = AverageMeter()
        start_time = time.time()

        for batch_idx, ((in_data, labels), (ood_data, ood_labels)) in enumerate(zip(train_loader, ood_loader)):
            
            ood_data, ood_labels = ood_data.to(device), ood_labels.to(device)

            in_data = torch.cat([in_data[0], in_data[1], in_data[2]], dim=0)
            in_data, labels = in_data.to(device), labels.to(device)

            # forward:
            all_data = torch.cat([in_data, ood_data], dim=0)
            all_logits, p4 = model(all_data)
            
            N_in = in_data.shape[0]
            f_id_view = p4[0:N_in]
            id_logits = all_logits[0:N_in]
            ood_logits = all_logits[N_in:]

            f1, f2, f3 = torch.split(f_id_view, [args.batch_size, args.batch_size, args.batch_size], dim=0)
            ce_logits, _, _ = torch.split(id_logits, [args.batch_size, args.batch_size, args.batch_size], dim=0)

            # ISAC: implicit semantic augmentation contrastive loss
            contrast_logits1 = criterion_scl(f2, labels, args=args)
            contrast_logits2 = criterion_scl(f3, labels, args=args)
            isac_loss = (criterion_ce(contrast_logits1, labels) + criterion_ce(contrast_logits2, labels))/2 

            # TLA: Temperature scaling-based logit adjustment loss
            tla_loss = criterion_ce(ce_logits/args.tem_scale, labels)

            # OE: Outlier exposure loss
            oe_loss = model.oe_loss_fn(ood_logits)

            # total loss:
            loss =  isac_loss + args.Lambda2 * tla_loss + args.Lambda1 * oe_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f ( %.4f, %.4f, %.4f)' % (epoch, batch_idx, loss.item(), isac_loss.item(), oe_loss.item(), tla_loss.item())#, tail_loss.item()) 
                train_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '  |  ' + train_str
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()
        model.eval()

        test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
        preds_list, labels_list = [], []
        #test
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                logits, _ = model(data)
                pred = logits.argmax(dim=1, keepdim=True)
                loss = F.cross_entropy(logits, labels)
                test_acc_meter.append((logits.data.max(1)[1] == labels).float().mean().item())
                test_loss_meter.append(loss.item())
                preds_list.append(pred)
                labels_list.append(labels)
                

        labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()
        preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
        overall_acc= (preds == labels).sum().item() / len(labels)
        f1 = f1_score(labels, preds, average='macro')
        many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)
        val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | time %.2f' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, time.time()-start_time) 
        val_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '  |  ' + val_str
        print(val_str)
        fp_val.write(val_str + '\n')
        fp_val.flush()

        test_clean_losses.append(test_loss_meter.avg)
        f1s.append(f1)
        overall_accs.append(overall_acc)
        many_accs.append(many_acc)
        median_accs.append(median_acc)
        low_accs.append(low_acc)

        # save curves:
        training_losses.append(training_loss_meter.avg)
        plt.plot(training_losses, 'b', label='training_losses')
        plt.plot(test_clean_losses, 'g', label='test_clean_losses')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'losses.png'))
        plt.close()

        plt.plot(overall_accs, 'm', label='overall_accs')
        if args.imbalance_ratio < 1:
            plt.plot(many_accs, 'r', label='many_accs')
            plt.plot(median_accs, 'g', label='median_accs')
            plt.plot(low_accs, 'b', label='low_accs')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'test_accs.png'))
        plt.close()

        plt.plot(f1s, 'm', label='f1s')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'test_f1s.png'))
        plt.close()

        # save pth:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch, 
            'training_losses': training_losses, 
            'test_clean_losses': test_clean_losses, 
            'f1s': f1s, 
            'overall_accs': overall_accs, 
            'many_accs': many_accs, 
            'median_accs': median_accs, 
            'low_accs': low_accs, 
            }, 
            os.path.join(save_dir, 'latest.pth'))
        
if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # mkdirs:
    save_dir, dataset_str = create_save_path()
    args.save_dir = save_dir
    args.dataset_str = dataset_str
    
    # intialize device:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    torch.backends.cudnn.benchmark = True
    
    train(args)
