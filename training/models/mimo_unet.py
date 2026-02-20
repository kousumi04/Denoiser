from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MIMOUNet(nn.Module):
    """
    Practical MIMO-UNet-style deblurring model with 3 output scales:
      quarter, half, full.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 48):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.enc3 = ConvBlock(c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.out_q = nn.Conv2d(c3, 3, 3, padding=1)
        self.out_h = nn.Conv2d(c2, 3, 3, padding=1)
        self.out_f = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        u2 = self.up2(e3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Residual prediction at each scale.
        # Use explicit sizes to avoid odd-dimension rounding mismatches.
        x_half = F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        x_quarter = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)

        pred_q = x_quarter + self.out_q(e3)
        pred_h = x_half + self.out_h(d2)
        pred_f = x + self.out_f(d1)

        return pred_q, pred_h, pred_f
