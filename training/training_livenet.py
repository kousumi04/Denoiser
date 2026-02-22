from __future__ import annotations

import argparse
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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

from data_pipeline.livenet_datasets import LiveNetDataset  # noqa: E402
from training.models.livenet import LiveNet  # noqa: E402


@dataclass
class TrainConfig:
    lol_root: Path | None = None
    lolv2_root: Path | None = None
    sice_root: Path | None = None
    img_size: int = 256
    exposure_target: float = 0.60
    exposure_tol: float = 0.05

    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_ratio: float = 0.1
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    best_name: str = "livenet.pth"
    final_name: str = "livenet_final.pth"


def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def auto_detect_dataset_roots() -> tuple[Path | None, Path | None, Path | None]:
    datasets_root = PROJECT_ROOT / "data_pipeline" / "datasets"
    lol_root = _first_existing_dir([datasets_root / "LOL_dataset" / "our485", datasets_root / "LOL_dataset"])
    lolv2_root = _first_existing_dir(
        [
            datasets_root / "Low_light_dataset(LOL_V2)" / "Real_captured" / "Train",
            datasets_root / "Low_light_dataset(LOL_V2)" / "Synthetic" / "Train",
            datasets_root / "Low_light_dataset(LOL_V2)",
        ]
    )
    return lol_root, lolv2_root, None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiveNet teacher on low-light paired data.")
    parser.add_argument("--lol-root", type=str, default=None)
    parser.add_argument("--lolv2-root", type=str, default=None)
    parser.add_argument("--sice-root", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    cfg: TrainConfig,
    stage: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=stage, leave=False)
    for inp, target in pbar:
        inp = inp.to(cfg.device, non_blocking=True)
        target = target.to(cfg.device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(cfg.use_amp, cfg.device):
            out = model(inp)
            loss = torch.mean(torch.abs(out - target))
        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.item())
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        lol_root=Path(args.lol_root) if args.lol_root else None,
        lolv2_root=Path(args.lolv2_root) if args.lolv2_root else None,
        sice_root=Path(args.sice_root) if args.sice_root else None,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_ratio=args.val_ratio,
        device=args.device,
        use_amp=not args.disable_amp,
    )

    if cfg.lol_root is None and cfg.lolv2_root is None and cfg.sice_root is None:
        cfg.lol_root, cfg.lolv2_root, cfg.sice_root = auto_detect_dataset_roots()
        print(f"Auto-detected roots: lol={cfg.lol_root}, lolv2={cfg.lolv2_root}, sice={cfg.sice_root}")
    if cfg.lol_root is None and cfg.lolv2_root is None and cfg.sice_root is None:
        raise ValueError("Provide at least one dataset root for LiveNet training.")

    set_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ds = LiveNetDataset(
        lol_root=str(cfg.lol_root) if cfg.lol_root else None,
        lolv2_root=str(cfg.lolv2_root) if cfg.lolv2_root else None,
        sice_root=str(cfg.sice_root) if cfg.sice_root else None,
        img_size=cfg.img_size,
        exposure_target=cfg.exposure_target,
        exposure_tol=cfg.exposure_tol,
    )
    if len(ds) < 2:
        raise RuntimeError("Need at least 2 samples for LiveNet train/val.")

    val_len = max(1, int(len(ds) * cfg.val_ratio))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        **_loader_extra_kwargs(cfg.num_workers),
    )
    val_workers = max(1, cfg.num_workers // 2)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        **_loader_extra_kwargs(val_workers),
    )

    model = LiveNet().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = _make_grad_scaler(cfg.use_amp, cfg.device)

    best_val = float("inf")
    best_path = cfg.checkpoint_dir / cfg.best_name
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, cfg, stage=f"train [{epoch}/{cfg.epochs}]")
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, scaler, cfg, stage=f"val   [{epoch}/{cfg.epochs}]")
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best LiveNet: {best_path}")

    final_path = cfg.checkpoint_dir / cfg.final_name
    torch.save(model.state_dict(), final_path)
    print("Training complete.")
    print(f"Best LiveNet: {best_path}")
    print(f"Final LiveNet: {final_path}")


if __name__ == "__main__":
    main()
