import argparse
import numpy as np
import torch

from mmseg import __version__
from mmseg.models.segmentors import PolypSegmentation as UNet
from thop import profile
from thop import clever_format

def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="b3")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument(
        "--test_path", type=str, default="./data/Datasets", help="path to dataset"
    )
    parser.add_argument(
        "--init_trainsize", type=str, default=352, help="path to dataset"
    )
    parser.add_argument("--train_save", type=str, default="polyp-seg")
    args = parser.parse_args()

    device = torch.device("cpu")

    model = UNet(
        backbone=dict(
            type="MSCAN",
            # depths=[2, 2, 4, 2],
            # depths=[3, 3, 12, 3],
            # depths=[3, 5, 27, 3],
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
        pretrained="pretrained/mscan_b.pth",
    ).to(device)

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    CalParams(model, torch.zeros(1, 3, 352, 352))
