from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.livenet import LiveNet  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "inference" / "outputs"
DEFAULT_CKPT_FINAL = PROJECT_ROOT / "checkpoints" / "livenet_final.pth"
DEFAULT_CKPT_BEST = PROJECT_ROOT / "checkpoints" / "livenet.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIPE LiveNet inference on a local image.")
    parser.add_argument("input_path", nargs="+", help="Input image path (e.g., img.png)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LiveNet checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _extract_state_dict(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v
        return cleaned
    return state


def _load_torch(path: Path, device: str):
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(path, **load_kwargs)


def _resolve_checkpoint(user_ckpt: str | None) -> Path:
    if user_ckpt is not None:
        ckpt = Path(user_ckpt)
        if ckpt.is_file():
            return ckpt
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if DEFAULT_CKPT_FINAL.is_file():
        return DEFAULT_CKPT_FINAL
    if DEFAULT_CKPT_BEST.is_file():
        return DEFAULT_CKPT_BEST

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    candidates = sorted(ckpt_dir.glob("livenet*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No LiveNet checkpoint found in AIPE/checkpoints/")


def _resolve_input(input_arg: str) -> Path:
    raw = Path(input_arg)
    if raw.is_file():
        return raw
    in_inference = PROJECT_ROOT / "inference" / input_arg
    if in_inference.is_file():
        return in_inference
    raise FileNotFoundError(f"Input image not found: {input_arg}")


def _read_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _save_image(img_tensor: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(out_path)


def _default_output_path(input_path: Path) -> Path:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{input_path.stem}_aipe_enhanced.png"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    input_arg = " ".join(args.input_path).strip()
    input_path = _resolve_input(input_arg)
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    out_path = Path(args.output) if args.output else _default_output_path(input_path)

    model = LiveNet().to(args.device)
    state = _load_torch(ckpt_path, args.device)
    model.load_state_dict(_extract_state_dict(state), strict=True)
    model.eval()

    inp = _read_image(input_path).to(args.device)
    with torch.no_grad():
        out = model(inp)

        # Reduce enhancement strength by blending more of the original input.
        out = 0.45 * out + 0.55 * inp

        # Edge-aware chroma denoise (targets orange/blue speckles while preserving structure).
        y = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        cb = out[:, 2:3] - y
        cr = out[:, 0:1] - y
        cb_s = F.avg_pool2d(cb, kernel_size=3, stride=1, padding=1)
        cr_s = F.avg_pool2d(cr, kernel_size=3, stride=1, padding=1)
        edge = (y - F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)).abs()
        edge_norm = edge / (edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
        flat_mask = torch.clamp(1.0 - 2.0 * edge_norm, 0.0, 1.0)
        denoise_mix = 0.65 * flat_mask
        cb = cb * (1.0 - denoise_mix) + cb_s * denoise_mix
        cr = cr * (1.0 - denoise_mix) + cr_s * denoise_mix
        r = y + cr
        b = y + cb
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        out = torch.cat([r, g, b], dim=1)

        # Decrease contrast by compressing values toward mid-gray.
        out = 0.90 * out + 0.20 * 0.5

        # Compute overall output brightness.
        brightness = out.mean().item()

        # Select adaptive multiplicative gain based on brightness.
        if brightness < 0.2:
            gain = 1.04
        elif brightness < 0.4:
            gain = 1.00
        else:
            gain = 0.97

        # Apply adaptive brightness scaling before clamping.
        out = gain * out

        # Clamp to valid image range.
        out = torch.clamp(out, 0.0, 1.0)

        # Debug info for applied gain.
        print(f"Adaptive brightness gain: {gain:.2f}")

    _save_image(out, out_path)
    print(f"AIPE checkpoint: {ckpt_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
