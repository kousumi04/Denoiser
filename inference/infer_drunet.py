# infer_drunet.py
# Complete runnable inference script

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.models.drunet import DRUNet
except ModuleNotFoundError:
    from training.models.drunet import DRUNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "DRUNET_DENOISER.pt"


def load_model(ckpt_path: Path) -> DRUNet:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = DRUNet().to(DEVICE)
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)

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
    model.eval()
    return model


def estimate_sigma(img_rgb: np.ndarray) -> float:
    gray = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    med = np.median(gray)
    mad = np.median(np.abs(gray - med))
    sigma = 1.4826 * mad
    return float(np.clip(sigma, 0.01, 0.25))


def read_image_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read input image: {path}")

    # grayscale -> 3-channel automatically
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def save_rgb(path: Path, img_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (np.clip(img_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out_bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="Manual sigma in [0,1]")
    args = parser.parse_args()

    input_path = Path(args.input)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()

    model = load_model(ckpt_path)
    img = read_image_any(input_path)

    inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    sigma_val = float(np.clip(args.sigma, 0.0, 1.0)) if args.sigma is not None else estimate_sigma(img)
    sigma = torch.tensor([[sigma_val]], dtype=inp.dtype, device=inp.device)

    with torch.no_grad():
        out = model(inp, sigma=sigma)
        out = torch.clamp(out, 0.0, 1.0)

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_denoised{input_path.suffix}")
    save_rgb(output_path, out_np)

    print(f"Input: {input_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sigma: {sigma_val:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
