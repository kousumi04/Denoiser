from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.fbcnn import FBCNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_STRENGTH_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def load_model(ckpt_path: Path) -> FBCNN:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = FBCNN().to(DEVICE)

    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def read_image(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    return tensor, (h, w)


def save_image(path: Path, tensor: torch.Tensor, size_hw: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = size_hw
    out = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    out = out[:h, :w]
    out = (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out).save(path)


def _to_hwc_np01(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()


def jpeg_blockiness_score(arr: np.ndarray) -> float:
    """Lower is better. Measures discontinuity at 8x8 JPEG block boundaries."""
    y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    h, w = y.shape
    vertical = list(range(8, w, 8))
    horizontal = list(range(8, h, 8))
    if not vertical or not horizontal:
        return 0.0

    v_boundary = np.mean([np.mean(np.abs(y[:, i] - y[:, i - 1])) for i in vertical])
    h_boundary = np.mean([np.mean(np.abs(y[i, :] - y[i - 1, :])) for i in horizontal])
    v_inner = np.mean([np.mean(np.abs(y[:, i + 1] - y[:, i])) for i in vertical if i + 1 < w])
    h_inner = np.mean([np.mean(np.abs(y[i + 1, :] - y[i, :])) for i in horizontal if i + 1 < h])
    return float((v_boundary + h_boundary) - (v_inner + h_inner))


def classical_deblock(arr: np.ndarray, severity: float) -> np.ndarray:
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    h_val = int(np.clip(6.0 + 12.0 * severity, 3.0, 18.0))
    ycc[:, :, 0] = cv2.fastNlMeansDenoising(ycc[:, :, 0], None, h=h_val, templateWindowSize=7, searchWindowSize=21)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    sigma_color = float(np.clip(15.0 + 40.0 * severity, 10.0, 55.0))
    out = cv2.bilateralFilter(out, d=5, sigmaColor=sigma_color, sigmaSpace=7.0)
    return out.astype(np.float32) / 255.0


def resolve_checkpoint_path(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    candidates = sorted(
        DEFAULT_CKPT_DIR.glob("fbcnn_epoch_*.pth"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint provided and no fbcnn_epoch_*.pth found in: {DEFAULT_CKPT_DIR}"
        )
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FBCNN JPEG artifact removal inference")
    parser.add_argument("input", nargs="?", default=None, help="Input image path")
    parser.add_argument("--input", dest="input_flag", type=str, default=None, help="Input image path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Checkpoint path (default: latest fbcnn_epoch_*.pth in {DEFAULT_CKPT_DIR})",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Manual residual strength; if omitted, auto-select from candidate strengths",
    )
    parser.add_argument(
        "--no-classical-fallback",
        action="store_true",
        help="Disable classical deblocking fallback when JPEG artifacts remain strong",
    )
    args = parser.parse_args()

    if args.input and args.input_flag and args.input != args.input_flag:
        parser.error("Conflicting input values provided via positional input and --input.")

    args.input = args.input_flag or args.input
    if not args.input:
        parser.error("Input image is required. Use positional input or --input.")

    return args


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    ckpt_path = resolve_checkpoint_path(args.checkpoint)

    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_jpeg_restored{input_path.suffix}")
    )

    model = load_model(ckpt_path)

    inp, size_hw = read_image(input_path)
    inp = inp.to(DEVICE)

    with torch.no_grad():
        base = model(inp)

    inp_np = _to_hwc_np01(inp)
    input_blockiness = jpeg_blockiness_score(inp_np)

    if args.strength is None:
        strengths = DEFAULT_STRENGTH_CANDIDATES
    else:
        strengths = [float(args.strength)]

    best_pred = None
    best_strength = None
    best_score = float("inf")

    for strength in strengths:
        pred = torch.clamp(inp + strength * (base - inp), 0.0, 1.0)
        pred_np = _to_hwc_np01(pred)
        blockiness = jpeg_blockiness_score(pred_np)
        mean_abs_delta = float((pred - inp).abs().mean().item() * 255.0)

        # Prefer lower blockiness, but avoid nearly-no-change or overly aggressive edits.
        score = blockiness
        if mean_abs_delta < 1.0:
            score += 0.01 * (1.0 - mean_abs_delta)
        if mean_abs_delta > 12.0:
            score += 0.001 * (mean_abs_delta - 12.0)

        if score < best_score:
            best_score = score
            best_pred = pred
            best_strength = strength

    assert best_pred is not None
    pred = best_pred
    selected_mode = f"fbcnn(strength={best_strength})"
    selected_blockiness = jpeg_blockiness_score(_to_hwc_np01(pred))

    if not args.no_classical_fallback and selected_blockiness >= input_blockiness - 0.002:
        severity = float(np.clip(input_blockiness * 20.0, 0.0, 1.0))
        classical_np = classical_deblock(inp_np, severity=severity)
        classical_tensor = torch.from_numpy(classical_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        classical_blockiness = jpeg_blockiness_score(classical_np)
        if classical_blockiness < selected_blockiness:
            pred = classical_tensor
            selected_mode = "classical_deblock_fallback"
            selected_blockiness = classical_blockiness

    save_image(output_path, pred, size_hw)

    mean_abs_delta_255 = float((pred - inp).abs().mean().item() * 255.0)

    print(f"Input: {input_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Mode: {selected_mode}")
    print(f"Input blockiness: {input_blockiness:.6f}")
    print(f"Output blockiness: {selected_blockiness:.6f}")
    print(f"Mean abs change: {mean_abs_delta_255:.3f}/255")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
