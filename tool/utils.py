import logging
from datasets.cityscapes import CityScapes
from datasets.cambridgeVideo import CamVid
from datasets.kitti import KITTI
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from torch.optim.lr_scheduler import _LRScheduler
import pickle as pkl
from glob import glob
import torch.nn as nn


def create_logger(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def read_pkl(pkl_name):
    with open(pkl_name, "rb") as f:
        data = pkl.load(f)
    return data


def save_pkl(pkl_file, pkl_name):
    with open(pkl_name, "wb") as f:
        pkl.dump(pkl_file, f)


def used_classes(set_name):
    if set_name == "cs":
        from datasets.cityscapes import index_label
    elif set_name == "cv":
        from datasets.cambridgeVideo import index_label
    elif set_name == "kitti":
        from datasets.kitti import index_label
    else:
        raise NotImplementedError

    return index_label


def define_dataset(set_name, mode, dataset_path, used_class, fully_supervised=False):
    if set_name == "cs":
        dataset = CityScapes(mode, dataset_path, used_class, fully_supervised)
    elif set_name == "cv":
        dataset = CamVid(mode, dataset_path, used_class, fully_supervised)
    elif set_name == 'kitti':
        dataset = KITTI(mode, dataset_path, used_class, fully_supervised)
    else:
        raise TypeError("Not exist dataset")
    return dataset


def get_dataloader(dataset, batch_size, num_workers, shuffle_):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_,
        drop_last=len(dataset) % batch_size == 1
    )

    return dataloader


def get_optimizer(model, case_="cs"):
    if case_ == "cs":
        optimizer_params = {
            "lr": 1e-5,
            "betas": (0.9, 0.999),
            "weight_decay": 2e-4,
            "eps": 1e-7
        }
    if case_ == "cv":
        optimizer_params = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 2e-4,
            "eps": 1e-7
        }
    if case_ == "kitti":
        optimizer_params = {
            "lr": 1e-5,
            "betas": (0.9, 0.999),
            "weight_decay": 2e-4,
            "eps": 1e-7
        }
    list_params = [{'params': model.backbone.parameters(),
                    'lr': optimizer_params['lr'] / 10,
                    'weight_decay': optimizer_params['weight_decay']}]

    list_params += [{'params': model.aspp.parameters(),
                     'lr': optimizer_params['lr'],
                     'weight_decay': optimizer_params['weight_decay']}]

    list_params += [{'params': model.low_level_conv.parameters(),
                     'lr': optimizer_params['lr'],
                     'weight_decay': optimizer_params['weight_decay']}]

    list_params += [{'params': model.seg_head.parameters(),
                     'lr': optimizer_params['lr'],
                     'weight_decay': optimizer_params['weight_decay']}]

    optimizer = Adam(list_params)

    return optimizer


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]


def get_lr_scheduler(n_epochs, optimizer, iters_per_epoch=-1, lr_scheduler_type="Poly"):
    if lr_scheduler_type == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    elif lr_scheduler_type == "Poly":
        lr_scheduler = Poly(optimizer, n_epochs, iters_per_epoch)
    return lr_scheduler


import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls_ = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls_)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        recall = np.diag(hist) / hist.sum(axis=0)
        f1 = 2 * (acc_cls_ * recall) / (acc_cls_ + recall)

        return (
            {
                "Pixel Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iu,
                "IoU": iu,
                "hist": hist,
                "Acc": acc_cls_,
                "f1": f1
            }
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def find_best_weights(root_path, label_name):
    best_weights_list = glob(f"{root_path}/*/{label_name}/best_epoch_model.pt")

    if len(best_weights_list) == 0:
        return None

    def find_the_int_num(txt_string):
        return list(map(int, txt_string.split("/")[-3][:-3]))[0]

    if os.path.exists(f"{root_path}/semi/{label_name}/best_epoch_model.pt"):
        return f"{root_path}/semi/{label_name}/best_epoch_model.pt"
    else:
        best_weights_list.sort(key=find_the_int_num)
        return best_weights_list[-1]


def custom_focal_loss(output, target, mask, alpha=0.25, gamma=2.0):
    """Custom focal-like loss for heatmap regression."""
    # Assuming output and target are the predicted and GT heatmaps, respectively
    bce_loss = nn.BCELoss(reduction='none')
    loss_per_element = bce_loss(output, target)

    # Apply mask
    loss_per_element = loss_per_element * mask

    # Apply focal mechanism: emphasize larger errors
    focal_factor = (loss_per_element.detach() + 1e-6) ** gamma  # detach to prevent this from affecting gradients
    focal_loss = alpha * focal_factor * loss_per_element

    return focal_loss.sum()