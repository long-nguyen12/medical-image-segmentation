import argparse
import os
from datetime import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.autograd import Variable

from mmseg import __version__
from mmseg.models.segmentors import PolypSegmentation as UNet
from schedulers import WarmupPolyLR
from utils import AvgMeter, clip_gradient
from val import inference


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, transform=None, color_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        if self.color_transform:
            augmented = self.color_transform(image=image)
            image = augmented["image"]

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)


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
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument(
        "--init_trainsize", type=int, default=352, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/dataset/",
        help="path to train dataset",
    )
    parser.add_argument("--train_save", type=str, default="polyp-seg-s")
    args = parser.parse_args()

    epochs = args.num_epochs
    ds = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]
    for _ds in ds:
        save_path = "snapshots/{}/{}/".format(args.train_save, _ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            print("Save path existed")

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
                A.Transpose(p=0.5),
                A.Affine(
                    scale=(0.5, 1.5),
                    translate_percent=(-0.125, 0.125),
                    rotate=(-180, 180),
                    shear=(-22.5, 22),
                    always_apply=True,
                ),
            ]
        )
        color_transform = A.Compose(
            [
                A.ColorJitter(
                    brightness=(0.6, 1.6),
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.01,
                    always_apply=True,
                ),
            ]
        )

        train_dataset = Dataset(
            train_img_paths,
            train_mask_paths,
            transform=transform,
            color_transform=color_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        print(len(train_loader))

        _total_step = len(train_loader)

        model = UNet(
            backbone=dict(
                type="MSCAN",
                depths=[3, 5, 27, 3],
                # depths=[3, 3, 12, 3],
                # drop_path_rate=0.1
            ),
            decode_head=dict(
                type="MLPPanHead",
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
            pretrained="pretrained/mscan_l.pth",
        ).cuda()

        # ---- flops and params ----
        params = model.parameters()
        optimizer = torch.optim.AdamW(
            params, args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        lr_scheduler = WarmupPolyLR(
            optimizer, 0.6, epochs * _total_step, _total_step * 10, 0.01
        )
        # optimizer = torch.optim.Adam(params, args.init_lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=len(train_loader) * epochs,
        #     eta_min=args.init_lr / 1000,
        # )

        start_epoch = 1

        best_iou = 0
        loss_record = AvgMeter()
        dice, iou = AvgMeter(), AvgMeter()

        print("#" * 20, "Start Training", "#" * 20)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                for i, pack in enumerate(train_loader, start=1):
                    if epoch <= 1:
                        optimizer.param_groups[0]["lr"] = (
                            (epoch * i) / (1.0 * _total_step) * args.init_lr
                        )
                    else:
                        lr_scheduler.step()

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
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

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
                    )
                )

            if epoch % 5 == 0:
                mean_iou, mean_dice, mean_precision, mean_recall = inference(
                    model, f"{args.train_path}/{_ds}/test/"
                )
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    ckpt_path = save_path + "base.pth"
                    print("[Saving Checkpoint:]", ckpt_path)
                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    }
                    torch.save(checkpoint, ckpt_path)

            if epoch == epochs:
                ckpt_path = save_path + "last.pth"
                print("[Saving Checkpoint:]", ckpt_path)
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                }
                torch.save(checkpoint, ckpt_path)
