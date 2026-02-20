from __future__ import annotations

import argparse
import copy
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

from data_pipeline.deblur_datasets import (  # noqa: E402
    DeblurPaths,
    MixedDeblurDataset,
    collect_div2k_pairs,
    collect_gopro_pairs,
    deblur_collate,
    split_pairs,
)
from training.models.mimo_unet import MIMOUNet  # noqa: E402


@dataclass
class TrainConfig:
    div2k_clean_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "Super_resolution(Div2k)"
    div2k_motion_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "Div2k_MotionBlur_Noisy"
    div2k_defocus_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "Div2k_DefocusBlur_Noisy"
    gopro_blur_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "gopro_deblur" / "blur"
    gopro_sharp_root: Path = PROJECT_ROOT / "data_pipeline" / "datasets" / "gopro_deblur" / "sharp"

    batch_size: int = 4
    num_workers: int = 4
    patch_size: int = 256
    epochs: int = 120
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    val_ratio_gopro: float = 0.1
    seed: int = 42
    use_amp: bool = True
    val_interval: int = 1
    early_stop_patience: int = 25

    ema_decay: float = 0.999
    use_ema: bool = True

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    best_checkpoint_name: str = "best_mimo_unet_deblur.pt"
    final_checkpoint_name: str = "mimo_unet_deblur_final.pt"
    resume: Path | None = None


def _loader_extra_kwargs(num_workers: int) -> dict:
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 4}


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
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 10.0 * torch.log10(1.0 / mse)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
    pred_e = nn.functional.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    tgt_e = nn.functional.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return torch.mean(torch.abs(pred_e - tgt_e))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to_model(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    paths = DeblurPaths(
        div2k_clean_root=cfg.div2k_clean_root,
        div2k_noisy_roots={
            "div2k_motion": cfg.div2k_motion_root,
            "div2k_defocus": cfg.div2k_defocus_root,
        },
        gopro_blur_root=cfg.gopro_blur_root,
        gopro_sharp_root=cfg.gopro_sharp_root,
    )

    div2k_train = collect_div2k_pairs(
        clean_root=paths.div2k_clean_root,
        noisy_roots=paths.div2k_noisy_roots,
        split_name="DIV2K_train_HR",
        verbose=True,
    )
    div2k_val = collect_div2k_pairs(
        clean_root=paths.div2k_clean_root,
        noisy_roots=paths.div2k_noisy_roots,
        split_name="DIV2K_valid_HR",
        verbose=True,
    )
    gopro_all = collect_gopro_pairs(paths.gopro_blur_root, paths.gopro_sharp_root, verbose=True)
    gopro_train, gopro_val = split_pairs(gopro_all, val_ratio=cfg.val_ratio_gopro, seed=cfg.seed)

    train_pairs = div2k_train + gopro_train
    val_pairs = div2k_val + gopro_val
    print(f"[build_loaders] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    train_ds = MixedDeblurDataset(
        train_pairs,
        patch_size=cfg.patch_size,
        training=True,
        augment=True,
    )
    val_ds = MixedDeblurDataset(
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
        collate_fn=deblur_collate,
        **_loader_extra_kwargs(cfg.num_workers),
    )
    val_workers = max(1, cfg.num_workers // 2)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=deblur_collate,
        **_loader_extra_kwargs(val_workers),
    )
    return train_loader, val_loader


def multi_scale_loss(pred_q: torch.Tensor, pred_h: torch.Tensor, pred_f: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    clean_h = nn.functional.interpolate(clean, scale_factor=0.5, mode="bilinear", align_corners=False)
    clean_q = nn.functional.interpolate(clean_h, scale_factor=0.5, mode="bilinear", align_corners=False)

    l_q = charbonnier_loss(pred_q, clean_q)
    l_h = charbonnier_loss(pred_h, clean_h)
    l_f = charbonnier_loss(pred_f, clean)
    l_e = edge_loss(pred_f, clean)

    return l_f + 0.5 * l_h + 0.25 * l_q + 0.05 * l_e


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    cfg: TrainConfig,
    stage: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    pbar = tqdm(loader, desc=stage, leave=False)
    for noisy, clean, _source in pbar:
        noisy = noisy.to(cfg.device, non_blocking=True)
        clean = clean.to(cfg.device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx(cfg.use_amp, cfg.device):
            pred_q, pred_h, pred_f = model(noisy)
            loss = multi_scale_loss(pred_q, pred_h, pred_f, clean)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            psnr = compute_psnr(torch.clamp(pred_f, 0.0, 1.0), clean)

        total_loss += float(loss.item())
        total_psnr += float(psnr.item())
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr.item():.2f}")

    return {"loss": total_loss / max(1, n), "psnr": total_psnr / max(1, n)}


def evaluate_model(model: nn.Module, val_loader: DataLoader, scaler: GradScaler, cfg: TrainConfig) -> dict[str, float]:
    with torch.no_grad():
        return run_epoch(model, val_loader, optimizer=None, scaler=scaler, cfg=cfg, stage="val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIMO-UNet deblurrer on DIV2K(defocus+motion)+GoPro.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--best-name", type=str, default="best_mimo_unet_deblur.pt")
    parser.add_argument("--final-name", type=str, default="mimo_unet_deblur_final.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_amp=not args.disable_amp,
        use_ema=not args.disable_ema,
        early_stop_patience=args.early_stop_patience,
        device=args.device,
        resume=Path(args.resume) if args.resume else None,
        best_checkpoint_name=args.best_name,
        final_checkpoint_name=args.final_name,
    )

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {cfg.device}")
    train_loader, val_loader = build_loaders(cfg)

    model = MIMOUNet().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.min_lr)
    scaler = _make_grad_scaler(cfg.use_amp, cfg.device)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    start_epoch = 1
    best_val_psnr = -1.0
    best_ckpt = cfg.checkpoint_dir / cfg.best_checkpoint_name
    bad_epochs = 0

    if cfg.resume is not None:
        if not cfg.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume}")
        try:
            state = torch.load(cfg.resume, map_location=cfg.device, weights_only=True)
        except TypeError:
            state = torch.load(cfg.resume, map_location=cfg.device)

        if "model_state_dict" in state:
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
            if ema is not None and "ema_state_dict" in state:
                ema.shadow = {k: v.detach().clone() for k, v in state["ema_state_dict"].items()}
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val_psnr = float(state.get("val_psnr", best_val_psnr))
            print(f"Resumed from {cfg.resume}, start_epoch={start_epoch}, best_val_psnr={best_val_psnr:.2f}")
        else:
            model.load_state_dict(state, strict=True)
            print(f"Loaded model-only weights from {cfg.resume}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, cfg, stage=f"train [{epoch}/{cfg.epochs}]")
        if ema is not None:
            ema.update(model)

        eval_model = model
        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.copy_to_model(eval_model)
            eval_model.eval()

        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val_stats = evaluate_model(eval_model, val_loader, scaler, cfg)
        else:
            val_stats = {"loss": float("nan"), "psnr": float("nan")}

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_stats['loss']:.5f} train_psnr={train_stats['psnr']:.2f} | "
            f"val_loss={val_stats['loss']:.5f} val_psnr={val_stats['psnr']:.2f} | lr={current_lr:.2e}"
        )

        if not np.isnan(val_stats["psnr"]) and val_stats["psnr"] > best_val_psnr:
            best_val_psnr = val_stats["psnr"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.shadow if ema is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "val_psnr": best_val_psnr,
                    "lr": current_lr,
                },
                best_ckpt,
            )
            print(f"  -> saved best checkpoint: {best_ckpt} (val_psnr={best_val_psnr:.2f})")
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val PSNR improvement for {bad_epochs} epochs).")
            break

    final_path = cfg.checkpoint_dir / cfg.final_checkpoint_name
    torch.save(model.state_dict(), final_path)

    print("\nTraining complete.")
    print(f"Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Final model state_dict: {final_path}")


if __name__ == "__main__":
    main()
