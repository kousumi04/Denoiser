# fine_tune_drunet.py
# Complete runnable fine-tuning script

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm

# DRUNet architecture
try:
    from models.drunet import DRUNet
except ModuleNotFoundError:
    from training.models.drunet import DRUNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# Config
# -----------------------------
PRETRAINED_CKPT = PROJECT_ROOT / "checkpoints" / "drunet_unified_final.pt"
SAVE_CKPT = PROJECT_ROOT / "checkpoints" / "DRUNET_DENOISER.pt"

# Noisy datasets
RGB_NOISY_ROOT = PROJECT_ROOT / "data_pipeline" / "datasets" / "Div2k_RandomCombo_Noisy"
GRAY_NOISY_ROOT = PROJECT_ROOT / "data_pipeline" / "datasets" / "Div2k_Gray_RandomCombo_Noisy"

# Clean root (paired by relative path)
CLEAN_ROOT = PROJECT_ROOT / "data_pipeline" / "datasets" / "Super_resolution(Div2k)"

BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-5
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 128
IMPULSE_AUG_PROB = 0.50
IMPULSE_DENSITY_MIN = 0.01
IMPULSE_DENSITY_MAX = 0.08

# Sampling ratio (normalized 60:30 => 2:1)
RGB_WEIGHT = 2.0
GRAY_WEIGHT = 1.0

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_rgb_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().contiguous().clone()


def paired_random_crop(noisy: np.ndarray, clean: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
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


def add_impulse_noise(img: np.ndarray, density: float) -> np.ndarray:
    """
    Salt-and-pepper style impulse corruption.
    A random subset of pixels is forced to 0 (pepper) or 1 (salt).
    """
    out = img.copy()
    h, w, _ = out.shape
    mask = np.random.rand(h, w) < density
    salt = np.random.rand(h, w) < 0.5
    out[mask & salt] = 1.0
    out[mask & (~salt)] = 0.0
    return out.astype(np.float32)


class PairedDiv2KDataset(Dataset):
    def __init__(self, noisy_root: Path, clean_root: Path, patch_size: int = 128):
        self.samples = []
        self.patch_size = patch_size
        noisy_root = Path(noisy_root)
        clean_root = Path(clean_root)

        if not noisy_root.is_dir():
            raise FileNotFoundError(f"Noisy root not found: {noisy_root}")
        if not clean_root.is_dir():
            raise FileNotFoundError(f"Clean root not found: {clean_root}")

        for p in noisy_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                rel = p.relative_to(noisy_root)
                c = clean_root / rel
                if c.is_file():
                    self.samples.append((p, c))

        if not self.samples:
            raise RuntimeError(f"No valid noisy-clean pairs found in {noisy_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.samples[idx]
        noisy = read_rgb_float(noisy_path)
        clean = read_rgb_float(clean_path)

        noisy, clean = paired_random_crop(noisy, clean, self.patch_size)
        if random.random() < IMPULSE_AUG_PROB:
            density = random.uniform(IMPULSE_DENSITY_MIN, IMPULSE_DENSITY_MAX)
            noisy = add_impulse_noise(noisy, density)

        noisy_t = to_tensor(noisy)
        clean_t = to_tensor(clean)

        # sigma hint for DRUNet
        sigma = torch.sqrt(torch.mean((noisy_t - clean_t) ** 2)).clamp(0.0, 1.0).unsqueeze(0)
        return noisy_t, clean_t, sigma


def build_loader():
    rgb_ds = PairedDiv2KDataset(RGB_NOISY_ROOT, CLEAN_ROOT, patch_size=PATCH_SIZE)
    gray_ds = PairedDiv2KDataset(GRAY_NOISY_ROOT, CLEAN_ROOT, patch_size=PATCH_SIZE)
    full_ds = ConcatDataset([rgb_ds, gray_ds])

    # Weighted sampling to enforce RGB:GRAY = 2:1 (equivalent to 60:30 normalized)
    weights = torch.zeros(len(full_ds), dtype=torch.float32)
    rgb_len = len(rgb_ds)
    gray_len = len(gray_ds)

    weights[:rgb_len] = RGB_WEIGHT / max(1, rgb_len)
    weights[rgb_len:] = GRAY_WEIGHT / max(1, gray_len)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(full_ds),   # one epoch length
        replacement=True
    )

    loader = DataLoader(
        full_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    print(f"RGB pairs: {rgb_len}, Gray pairs: {gray_len}, Total: {len(full_ds)}")
    return loader


def load_pretrained(model: nn.Module, ckpt: Path):
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    try:
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    model = DRUNet().to(DEVICE)
    load_pretrained(model, PRETRAINED_CKPT)

    loader = build_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for noisy, clean, sigma in pbar:
            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)
            sigma = sigma.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy, sigma=sigma)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = running / max(1, len(loader))
        print(f"Epoch {epoch}: avg L1 = {avg_loss:.6f}")

    SAVE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_CKPT)
    print(f"Saved fine-tuned model to: {SAVE_CKPT.resolve()}")


if __name__ == "__main__":
    main()
