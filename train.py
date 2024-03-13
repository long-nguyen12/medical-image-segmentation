import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
import albumentations as A
from mmseg import __version__
from mmseg.models.segmentors import UniPolyp as UNet
from val import inference
from schedulers import WarmupPolyLR


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        image = image.float()

        mask = mask.permute(2, 0, 1)

        return image, mask


epsilon = 1e-7


def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall * precision / (recall + precision - recall * precision + epsilon)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, help="epoch number")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
    parser.add_argument(
        "--init_trainsize", type=int, default=352, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/dataset",
        help="path to train dataset",
    )
    parser.add_argument("--train_save", type=str, default="mscan-base")
    args = parser.parse_args()

    ds = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]
    for _ds in ds:
        save_path = "snapshots/{}/{}/".format(args.train_save, _ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            print("Save path existed", flush=True)

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob("{}/{}/train/images/*".format(args.train_path, _ds))
        train_mask_paths = glob("{}/{}/train/masks/*".format(args.train_path, _ds))
        train_img_paths.sort()
        train_mask_paths.sort()

        transform = A.Compose(
            [
                A.Resize(height=352, width=352),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ColorJitter(
                    brightness=(0.6, 1.6),
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.01,
                    always_apply=True,
                ),
                A.Affine(
                    scale=(0.5, 1.5),
                    translate_percent=(-0.125, 0.125),
                    rotate=(-180, 180),
                    shear=(-22.5, 22),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

        train_dataset = Dataset(train_img_paths, train_mask_paths, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        print(len(train_loader), flush=True)

        total_step = len(train_loader)
        _total_step = len(train_loader)

        model = UNet(
            backbone=dict(
                type="MSCAN",
            ),
            decode_head=dict(
                type="MLPPanHead",
                # in_channels=[96, 192, 384, 768],
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=128,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=dict(type="BN", requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
                ),
            ),
            neck=None,
            auxiliary_head=None,
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
            pretrained="pretrained/mscan_b.pth",
        ).cuda()

        eps = 100
        # ---- flops and params ----
        params = model.parameters()
        optimizer = torch.optim.AdamW(
            params, args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
        )
        lr_scheduler = WarmupPolyLR(
            optimizer, 0.9, eps * _total_step, _total_step * 10, 0.01
        )
        start_epoch = 1

        best_iou = 0
        size_rates = [0.75, 1, 1.25]
        loss_record = AvgMeter()
        dice, iou = AvgMeter(), AvgMeter()

        print("#" * 20, "Start Training", "#" * 20, flush=True)
        for epoch in range(start_epoch, eps + 1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                for i, pack in enumerate(train_loader, start=1):
                    if epoch <= 1:
                        optimizer.param_groups[0]["lr"] = (
                            (epoch * i) / (1.0 * total_step) * args.init_lr
                        )
                    else:
                        lr_scheduler.step()

                    # for rate in size_rates:
                    optimizer.zero_grad()
                    # ---- data prepare ----
                    images, gts = pack
                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()

                    # ---- forward ----
                    map0, map4, map3, map2, map1 = model(images)
                    loss = (
                        structure_loss(map1, gts)
                        + structure_loss(map0, gts)
                        + structure_loss(map2, gts)
                        + structure_loss(map3, gts)
                        + structure_loss(map4, gts)
                    )

                    # ---- metrics ----
                    dice_score = dice_m(map0, gts)
                    iou_score = iou_m(map0, gts)
                    # ---- backward ----
                    loss.backward()
                    clip_gradient(optimizer, args.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    # if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

                # lr_scheduler.step()

                # ---- train visualization ----
                print(
                    "{} Training Epoch [{:03d}/{:03d}], "
                    "[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]".format(
                        datetime.now(),
                        epoch,
                        args.num_epochs,
                        loss_record.show(),
                        dice.show(),
                        iou.show(),
                    ),
                    flush=True,
                )

            if epoch % 5 == 0:
                mean_iou, mean_dice, mean_precision, mean_recall = inference(
                    model, f"{args.train_path}/{_ds}/test/"
                )
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    ckpt_path = save_path + "best.pth"
                    print("[Saving Checkpoint:]", ckpt_path, flush=True)
                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    }
                    torch.save(checkpoint, ckpt_path)

            if epoch == eps:
                ckpt_path = save_path + "last.pth"
                print("[Saving Checkpoint:]", ckpt_path, flush=True)
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                }
                torch.save(checkpoint, ckpt_path)
