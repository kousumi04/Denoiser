import torch
import torch.nn as nn

# ---------------------------------------------------
# LayerNorm2d
# ---------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------
# Memory-Efficient Attention (Channel Attention)
# ---------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = torch.sigmoid(self.conv1(x))
        return self.conv2(x * attn)


# ---------------------------------------------------
# Transformer Block
# ---------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------
# Lightweight Restormer
# ---------------------------------------------------

class Restormer(nn.Module):
    def __init__(self, dim=32, num_blocks=2):
        super().__init__()

        self.embedding = nn.Conv2d(3, dim, 3, padding=1)

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(num_blocks)]
        )

        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        inp = x
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.output(x)
        return x + inp
