import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_relu: bool = True,
        interpolate: bool = False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=True)
        return x
    
class FPABlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_mode="bicubic"):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == "bicubic":
            self.align_corners = True
        else:
            self.align_corners = False

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

        self.conv0 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)

        self.att_branch1 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.att_branch2 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.att_branch3 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=7, dilation=7)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        mid = self.conv0(x)

        att_branch1 = self.att_branch1(mid)
        att_branch2 = self.att_branch2(mid)
        att_branch3 = self.att_branch3(mid)

        att = att_branch1 + att_branch2 + att_branch3
        x = torch.mul(mid, att)
        # mid = self.mid(x)
        # x1 = self.down1(x)
        # x2 = self.down2(x1)
        # x3 = self.down3(x2)
        # x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        # x2 = self.conv2(x2)
        # x = x2 + x3
        # x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        # x1 = self.conv1(x1)
        # x = x + x1
        # x = F.interpolate(x, size=(h, w), **upscale_parameters)

        # x = torch.mul(x, mid)
        x = x + b1
        # return x
        return x


@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        # self.psp_modules = PPM(
        #     pool_scales,
        #     self.in_channels[-1],
        #     self.channels,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg,
        #     align_corners=self.align_corners)
        self.psp_modules = FPABlock(in_channels=self.in_channels[-1], out_channels=self.channels)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        # x = inputs[-1]
        # psp_outs = [x]
        # psp_outs.extend(self.psp_modules(x))
        # psp_outs = torch.cat(psp_outs, dim=1)
        # output = self.bottleneck(psp_outs)

        bottleneck = inputs[-1]
        output = self.psp_modules(bottleneck)  # 1/32

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bicubic',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bicubic',
                align_corners=self.align_corners)
        #fpn_outs = torch.cat(fpn_outs, dim=1)
        #output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(fpn_outs[-1])
        return output