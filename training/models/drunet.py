import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a local residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class DownBlock(nn.Module):
    """Feature extraction + strided downsampling."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.blocks(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsample + skip fusion + residual refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x


class DRUNet(nn.Module):
    """
    Lightweight DRUNet-style denoiser.

    We follow the common residual-denoising formulation:
      predicted_noise = f_theta([noisy, sigma_map])
      clean = noisy - predicted_noise

    where sigma_map is a (B,1,H,W) channel containing the noise level prior.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        num_blocks: int = 2,
    ):
        super().__init__()
        total_in = in_channels + 1  # +1 for sigma_map

        self.enc1 = DownBlock(total_in, base_channels, num_blocks)
        self.enc2 = DownBlock(base_channels, base_channels * 2, num_blocks)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4, num_blocks)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_channels * 8) for _ in range(num_blocks)],
        )

        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, num_blocks)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, num_blocks)
        self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels, num_blocks)

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, noisy: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            noisy: (B,3,H,W), normalized to [0,1]
            sigma: (B,) or (B,1) noise-level scalar in [0,1]. If None, zeros are used.
        """
        b, _, h, w = noisy.shape

        if sigma is None:
            sigma_map = torch.zeros((b, 1, h, w), dtype=noisy.dtype, device=noisy.device)
        else:
            if sigma.ndim == 1:
                sigma = sigma[:, None]
            sigma_map = sigma[:, :, None, None].expand(-1, -1, h, w)

        x = torch.cat([noisy, sigma_map], dim=1)

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)

        x = self.bottleneck(x)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        predicted_noise = self.out_conv(x)
        clean = noisy - predicted_noise
        return clean


# Backward compatibility alias if older code imported the typo name.
class D0RUNet(DRUNet):
    pass
