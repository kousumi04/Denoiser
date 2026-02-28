from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
JPEG_QUALITIES = [10, 20, 30, 40, 50, 60, 70, 80]


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"DIV2K root not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found under: {root}")
    return files


def _pil_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().float()


def _paired_random_crop(inp: np.ndarray, tgt: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h = min(inp.shape[0], tgt.shape[0])
    w = min(inp.shape[1], tgt.shape[1])
    inp = inp[:h, :w]
    tgt = tgt[:h, :w]

    if h < patch_size or w < patch_size:
        return inp, tgt

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        inp[top : top + patch_size, left : left + patch_size],
        tgt[top : top + patch_size, left : left + patch_size],
    )


class JPEGArtifactDataset(Dataset):
    """
    Creates synthetic JPEG-compressed inputs from clean DIV2K targets.
    Returns: input_tensor, target_tensor in [0,1], shape (3,H,W)
    """

    def __init__(self, clean_root: str | Path, patch_size: int = 128, training: bool = True):
        self.clean_root = Path(clean_root)
        self.image_paths = _list_images(self.clean_root)
        self.patch_size = int(patch_size)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _jpeg_compress(self, clean_img: Image.Image) -> Image.Image:
        quality = random.choice(JPEG_QUALITIES)
        buffer = io.BytesIO()

        # In-memory JPEG compression to synthesize artifacts.
        clean_img.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as tmp:
            jpeg_img = tmp.convert("RGB")
        return jpeg_img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.image_paths[idx]
        with Image.open(clean_path) as img:
            clean_pil = img.convert("RGB")

        jpeg_pil = self._jpeg_compress(clean_pil)

        target = _pil_to_float_np(clean_pil)
        inp = _pil_to_float_np(jpeg_pil)

        if self.training:
            inp, target = _paired_random_crop(inp, target, self.patch_size)

            if random.random() < 0.5:
                inp = np.flip(inp, axis=1)
                target = np.flip(target, axis=1)
            if random.random() < 0.5:
                inp = np.flip(inp, axis=0)
                target = np.flip(target, axis=0)

            inp = inp.copy()
            target = target.copy()

        return _to_tensor(inp), _to_tensor(target)
