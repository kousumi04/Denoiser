from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _read_rgb_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().clone()


def _paired_crop(noisy: np.ndarray, clean: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h = min(noisy.shape[0], clean.shape[0])
    w = min(noisy.shape[1], clean.shape[1])
    noisy = noisy[:h, :w]
    clean = clean[:h, :w]

    if h < patch_size or w < patch_size:
        noisy = cv2.resize(noisy, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        clean = cv2.resize(clean, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        return noisy, clean

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        noisy[top : top + patch_size, left : left + patch_size],
        clean[top : top + patch_size, left : left + patch_size],
    )


def _paired_augment(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    if k:
        noisy = np.rot90(noisy, k=k)
        clean = np.rot90(clean, k=k)

    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)
    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)
    return noisy.copy(), clean.copy()


@dataclass
class DeblurPaths:
    div2k_clean_root: Path
    div2k_noisy_roots: dict[str, Path]
    gopro_blur_root: Path
    gopro_sharp_root: Path


def collect_div2k_pairs(
    clean_root: Path,
    noisy_roots: dict[str, Path],
    split_name: str,
    verbose: bool = True,
) -> list[dict]:
    clean_split = clean_root / split_name
    if not clean_split.is_dir():
        raise FileNotFoundError(f"Missing clean split: {clean_split}")

    rel_files: list[Path] = []
    for root, _, files in os.walk(clean_split):
        root_path = Path(root)
        for name in files:
            p = root_path / name
            if p.suffix.lower() in VALID_EXTS:
                rel_files.append(p.relative_to(clean_split))
    rel_files = sorted(rel_files)

    pairs: list[dict] = []
    for rel in rel_files:
        clean_path = clean_split / rel
        for domain, noisy_root in noisy_roots.items():
            noisy_path = noisy_root / split_name / rel
            if noisy_path.is_file():
                pairs.append(
                    {
                        "source": domain,
                        "noisy_path": noisy_path,
                        "clean_path": clean_path,
                    }
                )

    if verbose:
        by_source: dict[str, int] = {}
        for p in pairs:
            by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        print(f"[collect_div2k_pairs] split={split_name} total={len(pairs)}")
        for k, v in sorted(by_source.items()):
            print(f"  - {k}: {v}")

    if not pairs:
        raise RuntimeError(f"No DIV2K pairs found for split={split_name}")
    return pairs


def collect_gopro_pairs(blur_root: Path, sharp_root: Path, verbose: bool = True) -> list[dict]:
    blur_files = sorted([p for p in blur_root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if not blur_files:
        raise RuntimeError(f"No blur files in {blur_root}")

    pairs: list[dict] = []
    for blur_path in blur_files:
        rel = blur_path.relative_to(blur_root)
        sharp_path = sharp_root / rel
        if sharp_path.is_file():
            pairs.append(
                {
                    "source": "gopro_motion",
                    "noisy_path": blur_path,
                    "clean_path": sharp_path,
                }
            )

    if verbose:
        print(f"[collect_gopro_pairs] total={len(pairs)}")
    if not pairs:
        raise RuntimeError("No GoPro blur/sharp pairs found")
    return pairs


def split_pairs(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    idxs = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_len = max(1, int(len(items) * val_ratio))
    val_set = set(idxs[:val_len])
    train_items = [items[i] for i in idxs[val_len:]]
    val_items = [items[i] for i in idxs[:val_len]]
    return train_items, val_items


class MixedDeblurDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict],
        patch_size: int = 256,
        training: bool = True,
        augment: bool = True,
    ):
        if not pairs:
            raise ValueError("pairs cannot be empty")
        self.pairs = pairs
        self.patch_size = patch_size
        self.training = training
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        noisy = _read_rgb_float(item["noisy_path"])
        clean = _read_rgb_float(item["clean_path"])

        if self.training:
            noisy, clean = _paired_crop(noisy, clean, self.patch_size)
            if self.augment:
                noisy, clean = _paired_augment(noisy, clean)
        else:
            h = min(noisy.shape[0], clean.shape[0])
            w = min(noisy.shape[1], clean.shape[1])
            noisy = noisy[:h, :w]
            clean = clean[:h, :w]

        return _to_tensor(noisy), _to_tensor(clean), item["source"]


def deblur_collate(batch):
    noisy, clean, source = zip(*batch)
    return (
        torch.stack([x.contiguous() for x in noisy], dim=0),
        torch.stack([x.contiguous() for x in clean], dim=0),
        list(source),
    )
