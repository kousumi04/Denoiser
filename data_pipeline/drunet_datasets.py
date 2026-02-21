import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_rgb_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def _augment_pair(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)
    if random.random() < 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)
    if random.random() < 0.5:
        noisy = np.rot90(noisy)
        clean = np.rot90(clean)
    return noisy.copy(), clean.copy()


def _paired_random_crop(noisy: np.ndarray, clean: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h = min(noisy.shape[0], clean.shape[0])
    w = min(noisy.shape[1], clean.shape[1])

    noisy = noisy[:h, :w]
    clean = clean[:h, :w]

    if h < patch_size or w < patch_size:
        # Keep deterministic interpolation path for tiny edge cases.
        noisy = cv2.resize(noisy, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        clean = cv2.resize(clean, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        return noisy, clean

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        noisy[top : top + patch_size, left : left + patch_size],
        clean[top : top + patch_size, left : left + patch_size],
    )


def _to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    # Use owned contiguous storage to avoid non-resizable storage issues in worker collation.
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().clone()


def drunet_collate(batch):
    """
    Stable collate for (noisy, clean, sigma, domain) tuples.
    Avoids default_collate's storage-resize path that can fail on some worker/platform combos.
    """
    noisy_list, clean_list, sigma_list, domain_list = zip(*batch)
    noisy = torch.stack([x.contiguous() for x in noisy_list], dim=0)
    clean = torch.stack([x.contiguous() for x in clean_list], dim=0)
    sigma = torch.stack([x.contiguous() for x in sigma_list], dim=0)
    domains = list(domain_list)
    return noisy, clean, sigma, domains


@dataclass
class Div2kPaths:
    clean_root: Path
    noisy_roots: dict[str, Path]


def collect_div2k_pairs(
    paths: Div2kPaths,
    split_name: str,
    verbose: bool = True,
) -> list[dict]:
    """
    Build paired samples by matching relative paths:
      clean_root / split_name / filename
      noisy_root / split_name / filename
    """
    clean_split_root = paths.clean_root / split_name
    if not clean_split_root.is_dir():
        raise FileNotFoundError(f"Clean split folder not found: {clean_split_root}")

    clean_rel_files: list[Path] = []
    for root, _, files in os.walk(clean_split_root):
        root_path = Path(root)
        for name in files:
            p = root_path / name
            if p.suffix.lower() in VALID_EXTS:
                clean_rel_files.append(p.relative_to(clean_split_root))

    clean_rel_files = sorted(clean_rel_files)
    if not clean_rel_files:
        raise RuntimeError(f"No clean images found under: {clean_split_root}")

    pairs: list[dict] = []
    for rel in clean_rel_files:
        clean_path = clean_split_root / rel
        for domain, noisy_root in paths.noisy_roots.items():
            noisy_path = noisy_root / split_name / rel
            if noisy_path.is_file():
                pairs.append(
                    {
                        "domain": domain,
                        "clean_path": clean_path,
                        "noisy_path": noisy_path,
                    }
                )

    if verbose:
        print(f"[collect_div2k_pairs] split={split_name}, pairs={len(pairs)}")
        by_domain: dict[str, int] = {}
        for item in pairs:
            by_domain[item["domain"]] = by_domain.get(item["domain"], 0) + 1
        for k, v in sorted(by_domain.items()):
            print(f"  - {k}: {v}")

    if not pairs:
        raise RuntimeError("No paired clean/noisy samples found. Check noisy dataset paths.")
    return pairs


class MixedDiv2KDenoiseDataset(Dataset):
    """
    Pairs multiple DIV2K noisy domains with a shared clean target.
    Returns:
      noisy (3,H,W), clean (3,H,W), sigma_hint (1,)

    sigma_hint formula:
      sigma_hat = sqrt(mean((noisy - clean)^2))
    This acts as a data-driven noise-level prior for DRUNet's sigma map.
    """

    def __init__(
        self,
        pairs: list[dict],
        patch_size: int = 128,
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
        entry = self.pairs[idx]
        noisy = _read_rgb_float(entry["noisy_path"])
        clean = _read_rgb_float(entry["clean_path"])

        if self.training:
            noisy, clean = _paired_random_crop(noisy, clean, self.patch_size)
            if self.augment:
                noisy, clean = _augment_pair(noisy, clean)
        else:
            h = min(noisy.shape[0], clean.shape[0])
            w = min(noisy.shape[1], clean.shape[1])
            noisy = noisy[:h, :w]
            clean = clean[:h, :w]

        diff = noisy - clean
        sigma_hat = float(np.sqrt(np.mean(diff * diff)))
        sigma_hat = max(0.0, min(1.0, sigma_hat))

        return (
            _to_chw_tensor(noisy),
            _to_chw_tensor(clean),
            torch.tensor([sigma_hat], dtype=torch.float32),
            entry["domain"],
        )


class SIDDPreprocessedDataset(Dataset):
    """
    Loads preprocessed .pt samples containing:
      {'noisy': CxHxW tensor, 'clean': CxHxW tensor}
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"SIDD root not found: {self.root_dir}")
        self.files = sorted(self.root_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found in: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        try:
            sample = torch.load(path, weights_only=True)
        except TypeError:
            sample = torch.load(path)

        noisy = sample["noisy"].float().contiguous().clone()
        clean = sample["clean"].float().contiguous().clone()

        # sigma_hat = sqrt(mean((noisy-clean)^2))
        sigma_hat = torch.sqrt(torch.mean((noisy - clean) ** 2)).clamp(0.0, 1.0)
        return noisy, clean, sigma_hat[None], "sidd"
