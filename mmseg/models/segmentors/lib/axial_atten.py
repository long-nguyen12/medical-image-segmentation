# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.conv_layer import Conv
from lib.self_attention import self_attn
import math


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3),stride = 1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx
    
class EMHA(nn.Module):
    def __init__(self, dim, head=1, sr_ratio=1):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio+1, sr_ratio, sr_ratio//2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = head > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(head, head, 1, 1)
            self.transform_norm = nn.InstanceNorm2d(head)

    def forward(self, x, H, W):
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x