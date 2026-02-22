from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiveNet(nn.Module):
    """
    Lightweight teacher network for low-light paired supervision.
    Returns enhanced RGB image for standalone use and feature extraction.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = _ConvBlock(3, base_ch)
        self.enc2 = _ConvBlock(base_ch, base_ch * 2)
        self.enc3 = _ConvBlock(base_ch * 2, base_ch * 2)
        self.dec1 = _ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.enc3(e2)
        d1 = self.dec1(self.up(b))
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.out(d1 + e1)
        return torch.clamp(out, 0.0, 1.0)
