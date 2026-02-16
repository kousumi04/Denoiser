import os
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "preprocessed_sidd", "train")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

BATCH_SIZE = 4          # Reduced for 8GB GPU
EPOCHS = 100
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================
# DATASET
# ==============================

class PreprocessedSIDD(Dataset):
    def __init__(self, root_dir):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ]
        if not self.files:
            raise RuntimeError(f"No .pt files found in: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=True)
        except TypeError:
            # Older torch versions do not support weights_only in torch.load.
            data = torch.load(self.files[idx])
        return data["noisy"], data["clean"]


# ==============================
# PSNR
# ==============================

def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ==============================
# IMPORT MODEL
# ==============================

try:
    from training.models.restormer import Restormer
except ModuleNotFoundError:
    # Support running as: python training/training_restormer.py
    from models.restormer import Restormer


# ==============================
# TRAIN LOOP
# ==============================

def train():

    dataset = PreprocessedSIDD(DATA_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = Restormer().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    use_amp = DEVICE == "cuda"
    try:
        scaler = GradScaler(device="cuda", enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    best_psnr = 0

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0
        epoch_psnr = 0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for noisy, clean in progress:

            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = autocast(device_type="cuda") if use_amp else nullcontext()
            with amp_ctx:
                output = model(noisy)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            with torch.no_grad():
                psnr = compute_psnr(output, clean)
                epoch_psnr += psnr.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{psnr.item():.2f}"
            )

            # -------------------------
            # CUDA Cache Cleaning
            # -------------------------
            del noisy, clean, output, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(loader)
        avg_psnr = epoch_psnr / len(loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB\n")

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_restormer.pth")
            )
            print("Best model saved.\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training Complete.")


if __name__ == "__main__":
    train()
