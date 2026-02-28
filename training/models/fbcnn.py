from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Simple residual block used in encoder and decoder stages."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(num_blocks)])
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv_in(x)
        feat = self.blocks(feat)
        down = self.down(feat)
        return feat, down


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd-sized inputs by cropping to shared size.
        if skip.shape[-2:] != x.shape[-2:]:
            h = min(skip.shape[-2], x.shape[-2])
            w = min(skip.shape[-1], x.shape[-1])
            skip = skip[:, :, :h, :w]
            x = x[:, :, :h, :w]

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x


class FBCNN(nn.Module):
    """
    Lightweight encoder-decoder JPEG artifact removal model.

    Input:  RGB image in [0,1], shape (N,3,H,W)
    Output: RGB image; clamp externally to [0,1] during inference
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = EncoderStage(3, c1, num_blocks=2)
        self.enc2 = EncoderStage(c1, c2, num_blocks=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
            ResidualBlock(c3),
            nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = DecoderStage(c2, c2, c2, num_blocks=2)
        self.dec1 = DecoderStage(c2, c1, c1, num_blocks=2)

        self.head = nn.Conv2d(c1, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x1 = self.enc1(x)
        s2, x2 = self.enc2(x1)

        x3 = self.bottleneck(x2)

        y2 = self.dec2(x3, s2)
        y1 = self.dec1(y2, s1)

        residual = self.head(y1)
        out = x + residual
        return out
