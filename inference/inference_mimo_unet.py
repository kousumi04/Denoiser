from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.mimo_unet import MIMOUNet  # noqa: E402


DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best_mimo_unet_deblur.pt"
DEFAULT_FINAL = PROJECT_ROOT / "checkpoints" / "mimo_unet_deblur_final.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "inference" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIMO-UNet image deblurring inference.")
    parser.add_argument("input_path", type=str, help="Input image path, e.g. myphoto.jpg")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--weights",
        type=str,
        default="model",
        choices=["model", "ema"],
        help="Choose checkpoint weights. 'model' is often sharper, 'ema' is often smoother.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for memory-safe inference")
    parser.add_argument("--tile-overlap", type=int, default=32, help="Overlap between neighboring tiles")
    return parser.parse_args()


def choose_checkpoint(user_ckpt: str | None) -> Path:
    if user_ckpt:
        ckpt = Path(user_ckpt)
    elif DEFAULT_CKPT.is_file():
        ckpt = DEFAULT_CKPT
    else:
        ckpt = DEFAULT_FINAL
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def load_model(ckpt_path: Path, device: str, weights: str = "model") -> MIMOUNet:
    model = MIMOUNet().to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and ("model_state_dict" in state or "ema_state_dict" in state):
        if weights == "ema" and state.get("ema_state_dict") is not None:
            state = state["ema_state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif state.get("ema_state_dict") is not None:
            state = state["ema_state_dict"]

    cleaned = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def to_image(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


@torch.no_grad()
def forward_full(model: MIMOUNet, x: torch.Tensor) -> torch.Tensor:
    _, _, out = model(x)
    return torch.clamp(out, 0.0, 1.0)


@torch.no_grad()
def forward_tiled(model: MIMOUNet, x: torch.Tensor, tile_size: int, tile_overlap: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if b != 1:
        raise ValueError("Tiled inference expects batch size 1.")

    tile = max(64, tile_size)
    overlap = max(0, min(tile_overlap, tile // 2))
    stride = tile - overlap
    if h <= tile and w <= tile:
        return forward_full(model, x)

    out = torch.zeros_like(x)
    norm = torch.zeros_like(x)

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))

    yy = torch.linspace(0, math.pi, tile, device=x.device)
    wx = (1.0 - torch.cos(yy)) * 0.5
    w2d = torch.outer(wx, wx).clamp_min(1e-6).view(1, 1, tile, tile)

    for y in ys:
        for x0 in xs:
            patch = x[:, :, y : y + tile, x0 : x0 + tile]
            ph, pw = patch.shape[-2:]
            if ph != tile or pw != tile:
                pad_h = tile - ph
                pad_w = tile - pw
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

            pred = forward_full(model, patch)
            pred = pred[:, :, :ph, :pw]
            weight = w2d[:, :, :ph, :pw]

            out[:, :, y : y + ph, x0 : x0 + pw] += pred * weight
            norm[:, :, y : y + ph, x0 : x0 + pw] += weight

    return torch.clamp(out / norm.clamp_min(1e-6), 0.0, 1.0)


def make_default_output(input_path: Path) -> Path:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{input_path.stem}_mimo_unet_deblurred.png"


def main() -> None:
    args = parse_args()
    inp_path = Path(args.input_path)
    ckpt_path = choose_checkpoint(args.checkpoint)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    model = load_model(ckpt_path, args.device, weights=args.weights)
    img = read_image(inp_path)
    x = to_tensor(img, args.device)

    with torch.no_grad():
        out = forward_tiled(model, x, tile_size=args.tile_size, tile_overlap=args.tile_overlap)

    out_path = Path(args.output) if args.output else make_default_output(inp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_image(out)).save(out_path)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
