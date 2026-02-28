from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.jpeg_dataset import JPEGArtifactDataset  # noqa: E402
from training.models.fbcnn import FBCNN  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DIV2K = PROJECT_ROOT / "data_pipeline" / "datasets" / "Super_resolution(Div2k)" / "DIV2K_train_HR"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_extra_kwargs(num_workers: int) -> dict:
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 2}


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Optional small SSIM loss term: loss = 1 - mean(SSIM)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=window_size, stride=1, padding=pad) - mu_x * mu_y

    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return 1.0 - ssim_map.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FBCNN-like JPEG artifact remover.")
    parser.add_argument("--div2k-root", type=str, default=str(DEFAULT_DIV2K), help="Path to clean DIV2K images")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=128, choices=[128, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ssim", action="store_true", help="Add small SSIM term")
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = JPEGArtifactDataset(clean_root=args.div2k_root, patch_size=args.patch_size, training=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
        **_loader_extra_kwargs(args.num_workers),
    )

    model = FBCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1_fn = nn.L1Loss()

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"train [{epoch}/{args.epochs}]", leave=False)
        for inp, target in pbar:
            inp = inp.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(inp)
            loss = l1_fn(pred, target)

            if args.use_ssim:
                loss = loss + args.ssim_weight * ssim_loss(pred, target)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(1, len(loader))
        ckpt_path = ckpt_dir / f"fbcnn_epoch_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": mean_loss,
            },
            ckpt_path,
        )

        print(f"Epoch {epoch}/{args.epochs} | loss={mean_loss:.6f} | saved={ckpt_path}")


if __name__ == "__main__":
    main()
