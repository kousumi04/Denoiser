import argparse
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
    _AMP_API = "torch.amp"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_API = "torch.cuda.amp"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.drunet_datasets import (  # noqa: E402
    Div2kPaths,
    MixedDiv2KDenoiseDataset,
    SIDDPreprocessedDataset,
    collect_div2k_pairs,
    drunet_collate,
)
from training.models.drunet import DRUNet  # noqa: E402


@dataclass
class TrainConfig:
    # Data
    div2k_clean_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "Super_resolution(Div2k)"
    noisy_root_base: Path = PROJECT_ROOT / "data_pipeline" / "datasets"
    sidd_preprocessed_root: Path = PROJECT_ROOT / "preprocessed_sidd" / "train"

    # Selected noisy domains requested by user
    noisy_domains: tuple[str, ...] = (
        "DIV2K_Gaussian_Noisy",
        "Div2k_JPEG_Noisy",
        "Div2k_LowLight_Noisy",
        "Div2k_Poisson_Noisy",
        "Div2k_RandomCombo_Noisy",
        "Div2k_Speckle_Noisy",
    )

    # Optimization
    batch_size: int = 4
    num_workers: int = 2
    total_epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patch_size: int = 128
    grad_clip_norm: float = 1.0
    seed: int = 42
    max_train_steps: int = 0
    max_val_steps: int = 0
    val_interval: int = 1

    # Mixed precision
    use_amp: bool = True

    # Bookkeeping
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_path: Path | None = None
    best_checkpoint_name: str = "best_drunet_unified.pt"
    final_checkpoint_name: str = "drunet_unified_final.pt"


def _loader_extra_kwargs(num_workers: int) -> dict:
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


def _make_grad_scaler(use_amp: bool, device: str) -> GradScaler:
    enabled = use_amp and device == "cuda"
    if _AMP_API == "torch.amp":
        try:
            return GradScaler(device="cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    return GradScaler(enabled=enabled)


def _autocast_ctx(use_amp: bool, device: str):
    if not (use_amp and device == "cuda"):
        return nullcontext()
    if _AMP_API == "torch.amp":
        return autocast(device_type="cuda")
    return autocast()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    PSNR formula for normalized images:
      MSE = mean((pred - target)^2)
      PSNR = 10 * log10(MAX_I^2 / MSE)
    With MAX_I = 1 for [0,1] tensors:
      PSNR = 10 * log10(1 / MSE)
    """
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier (smooth L1-like) loss:
      L = mean( sqrt((pred - target)^2 + eps^2) )
    """
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def build_div2k_paths(cfg: TrainConfig) -> Div2kPaths:
    noisy_roots = {}
    for folder_name in cfg.noisy_domains:
        root = cfg.noisy_root_base / folder_name
        if not root.is_dir():
            raise FileNotFoundError(f"Noisy dataset folder not found: {root}")
        noisy_roots[folder_name] = root
    return Div2kPaths(clean_root=cfg.div2k_clean_root, noisy_roots=noisy_roots)


def build_div2k_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    paths = build_div2k_paths(cfg)

    train_pairs = collect_div2k_pairs(paths, split_name="DIV2K_train_HR", verbose=True)
    val_pairs = collect_div2k_pairs(paths, split_name="DIV2K_valid_HR", verbose=True)

    train_ds = MixedDiv2KDenoiseDataset(
        train_pairs,
        patch_size=cfg.patch_size,
        training=True,
        augment=True,
    )
    val_ds = MixedDiv2KDenoiseDataset(
        val_pairs,
        patch_size=cfg.patch_size,
        training=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(cfg.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(max(1, cfg.num_workers // 2)),
    )
    return train_loader, val_loader


def build_sidd_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    sidd_ds = SIDDPreprocessedDataset(cfg.sidd_preprocessed_root)
    val_len = max(1, int(0.1 * len(sidd_ds)))
    train_len = len(sidd_ds) - val_len
    sidd_train, sidd_val = random_split(sidd_ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(
        sidd_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(cfg.num_workers),
    )
    val_loader = DataLoader(
        sidd_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(max(1, cfg.num_workers // 2)),
    )
    return train_loader, val_loader


def build_all_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """
    Build one unified train/val set by concatenating:
      1) DIV2K noisy domains (gaussian/jpeg/lowlight/poisson/randomcombo/speckle)
      2) Preprocessed SIDD patches
    """
    paths = build_div2k_paths(cfg)
    div2k_train_pairs = collect_div2k_pairs(paths, split_name="DIV2K_train_HR", verbose=True)
    div2k_val_pairs = collect_div2k_pairs(paths, split_name="DIV2K_valid_HR", verbose=True)
    # Use all DIV2K samples for training (train + valid splits together).
    div2k_all_pairs = div2k_train_pairs + div2k_val_pairs

    div2k_train_ds = MixedDiv2KDenoiseDataset(
        div2k_all_pairs,
        patch_size=cfg.patch_size,
        training=True,
        augment=True,
    )
    div2k_val_ds = MixedDiv2KDenoiseDataset(
        div2k_val_pairs,
        patch_size=cfg.patch_size,
        training=False,
        augment=False,
    )

    sidd_all_ds = SIDDPreprocessedDataset(cfg.sidd_preprocessed_root)
    sidd_val_len = max(1, int(0.1 * len(sidd_all_ds)))
    sidd_train_len = len(sidd_all_ds) - sidd_val_len
    sidd_train_ds, sidd_val_ds = random_split(
        sidd_all_ds,
        [sidd_train_len, sidd_val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    # Train on all SIDD samples by concatenating both partitions.
    sidd_all_train_ds = ConcatDataset([sidd_train_ds, sidd_val_ds])

    unified_train_ds = ConcatDataset([div2k_train_ds, sidd_all_train_ds])
    unified_val_ds = ConcatDataset([div2k_val_ds, sidd_val_ds])

    print(f"[build_all_loaders] unified_train={len(unified_train_ds)} unified_val={len(unified_val_ds)}")

    train_loader = DataLoader(
        unified_train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(cfg.num_workers),
    )
    val_workers = max(1, cfg.num_workers // 2)
    val_loader = DataLoader(
        unified_val_ds,
        # Validation contains mixed spatial sizes (full-res DIV2K + fixed-size SIDD patches),
        # so we keep batch_size=1 to avoid collation shape conflicts.
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=drunet_collate,
        **_loader_extra_kwargs(val_workers),
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
    stage_name: str,
    max_steps: int = 0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    pbar = tqdm(loader, desc=stage_name, leave=False)
    for step_idx, (noisy, clean, sigma, _domain) in enumerate(pbar, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        sigma = sigma.to(device, non_blocking=True).view(noisy.shape[0], 1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        amp_ctx = _autocast_ctx(use_amp=use_amp, device=device)
        with amp_ctx:
            pred = model(noisy, sigma=sigma)

            # Main objective:
            #   L_total = L1(pred, clean) + 0.1 * Charbonnier(pred, clean)
            # L1: mean(|pred-clean|)
            # Charbonnier: mean(sqrt((pred-clean)^2 + eps^2))
            l1 = torch.mean(torch.abs(pred - clean))
            ch = charbonnier_loss(pred, clean)
            loss = l1 + 0.1 * ch

        if is_train:
            scaler.scale(loss).backward()

            # Gradient clipping formula:
            #   g <- g * min(1, max_norm / ||g||_2)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            psnr = compute_psnr(torch.clamp(pred, 0.0, 1.0), clean)

        total_loss += float(loss.item())
        total_psnr += float(psnr.item())
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr.item():.2f}")

        if max_steps > 0 and step_idx >= max_steps:
            break

    return {
        "loss": total_loss / max(1, n),
        "psnr": total_psnr / max(1, n),
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    cfg: TrainConfig,
    ckpt_name: str,
    resume_path: Path | None = None,
) -> tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = _make_grad_scaler(use_amp=cfg.use_amp, device=cfg.device)

    best_val_psnr = -1.0
    start_epoch = 1
    ckpt_path = cfg.checkpoint_dir / ckpt_name

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            state = torch.load(resume_path, map_location=cfg.device, weights_only=True)
        except TypeError:
            state = torch.load(resume_path, map_location=cfg.device)

        # Supports both full training checkpoints and plain state_dict checkpoints.
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            if "scaler_state_dict" in state:
                try:
                    scaler.load_state_dict(state["scaler_state_dict"])
                except Exception:
                    pass
            if "epoch" in state:
                start_epoch = int(state["epoch"]) + 1
            if "val_psnr" in state:
                best_val_psnr = float(state["val_psnr"])
            print(f"Resumed from {resume_path} at epoch {start_epoch} (best_val_psnr={best_val_psnr:.2f})")
        else:
            model.load_state_dict(state, strict=True)
            print(f"Loaded model weights from {resume_path} (optimizer/scheduler reset)")

    if start_epoch > epochs:
        print(f"Start epoch {start_epoch} is greater than configured epochs {epochs}. Nothing to train.")
        return model, best_val_psnr

    for epoch in range(start_epoch, epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=cfg.device,
            use_amp=cfg.use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
            stage_name=f"train [{epoch}/{epochs}]",
            max_steps=cfg.max_train_steps,
        )
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scaler=scaler,
                    device=cfg.device,
                    use_amp=cfg.use_amp,
                    grad_clip_norm=cfg.grad_clip_norm,
                    stage_name=f"val   [{epoch}/{epochs}]",
                    max_steps=cfg.max_val_steps,
                )
        else:
            val_stats = {"loss": float("nan"), "psnr": float("nan")}

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if np.isnan(val_stats["psnr"]):
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_stats['loss']:.5f} train_psnr={train_stats['psnr']:.2f} | "
                f"val=skipped | lr={current_lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_stats['loss']:.5f} train_psnr={train_stats['psnr']:.2f} | "
                f"val_loss={val_stats['loss']:.5f} val_psnr={val_stats['psnr']:.2f} | lr={current_lr:.2e}"
            )

        if not np.isnan(val_stats["psnr"]) and val_stats["psnr"] > best_val_psnr:
            best_val_psnr = val_stats["psnr"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_psnr": best_val_psnr,
                    "lr": current_lr,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                ckpt_path,
            )
            print(f"  -> best checkpoint saved: {ckpt_path} (val_psnr={best_val_psnr:.2f})")

    return model, best_val_psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one unified DRUNet on all datasets together.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-steps", type=int, default=0, help="0 means full validation")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Optional path to .pt checkpoint to resume from")
    parser.add_argument("--best-name", type=str, default="best_drunet_unified.pt", help="Best checkpoint filename")
    parser.add_argument("--final-name", type=str, default="drunet_unified_final.pt", help="Final model filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        total_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        use_amp=not args.disable_amp,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        val_interval=args.val_interval,
        resume_path=Path(args.resume) if args.resume else None,
        best_checkpoint_name=args.best_name,
        final_checkpoint_name=args.final_name,
    )

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {cfg.device}")

    print("\nBuilding unified loaders (DIV2K noisy domains + SIDD)...")
    train_loader, val_loader = build_all_loaders(cfg)

    model = DRUNet().to(cfg.device)
    print("Training single unified DRUNet...")
    model, best_val_psnr = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.total_epochs,
        lr=cfg.lr,
        cfg=cfg,
        ckpt_name=cfg.best_checkpoint_name,
        resume_path=cfg.resume_path,
    )

    final_path = cfg.checkpoint_dir / cfg.final_checkpoint_name
    torch.save(model.state_dict(), final_path)

    print("\nTraining complete.")
    print(f"Best unified validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Best checkpoint: {cfg.checkpoint_dir / cfg.best_checkpoint_name}")
    print(f"Final model state_dict: {final_path}")
    print(f"Checkpoints saved in: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
