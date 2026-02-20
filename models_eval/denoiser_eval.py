# eval.py
import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DRUNet architecture
try:
    from models.drunet import DRUNet
except ModuleNotFoundError:
    from training.models.drunet import DRUNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "DRUNET_DENOISER.pt"
OUT_PATH = SCRIPT_DIR / "output_denoised.png"

# NIQE model files are not reliably hosted for current OpenCV contrib releases.
# Keep NIQE local-file path support, and fallback to BRISQUE online assets.
NIQE_MODEL_LOCAL = SCRIPT_DIR / ".niqe_assets" / "niqe_model_live.yml"
NIQE_RANGE_LOCAL = SCRIPT_DIR / ".niqe_assets" / "niqe_range_live.yml"

BRISQUE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_model_live.yml"
)
BRISQUE_RANGE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_range_live.yml"
)


def parse_args():
    parser = argparse.ArgumentParser(description="DRUNet eval with NIQE (no-reference).")
    parser.add_argument("input", type=str, help="Input image path (RGB or grayscale).")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to DRUNET_DENOISER.pt")
    parser.add_argument("--output", type=str, default=str(OUT_PATH), help="Path to save denoised output image")
    return parser.parse_args()


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


def read_image_rgb_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    # grayscale -> 3-channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def save_rgb(path: Path, img_rgb: np.ndarray):
    out = (np.clip(img_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out_bgr)


def ensure_brisque_files(base_dir: Path) -> tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    model_path = base_dir / "brisque_model_live.yml"
    range_path = base_dir / "brisque_range_live.yml"

    if not model_path.exists():
        urllib.request.urlretrieve(BRISQUE_MODEL_URL, str(model_path))
    if not range_path.exists():
        urllib.request.urlretrieve(BRISQUE_RANGE_URL, str(range_path))

    return model_path, range_path


def compute_niqe(img_rgb_float: np.ndarray, model_path: Path, range_path: Path) -> float:
    if not hasattr(cv2, "quality") or not hasattr(cv2.quality, "QualityNIQE_compute"):
        raise RuntimeError(
            "OpenCV NIQE API not found. Install opencv-contrib-python."
        )

    gray_u8 = cv2.cvtColor((img_rgb_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Returns scalar or tuple depending on OpenCV build.
    score = cv2.quality.QualityNIQE_compute(gray_u8, str(model_path), str(range_path))
    if isinstance(score, tuple) or isinstance(score, list):
        score = score[0]
    if isinstance(score, np.ndarray):
        score = float(score.ravel()[0])
    return float(score)


def compute_brisque(img_rgb_float: np.ndarray, model_path: Path, range_path: Path) -> float:
    if not hasattr(cv2, "quality") or not hasattr(cv2.quality, "QualityBRISQUE_compute"):
        raise RuntimeError("BRISQUE API not available")
    bgr_u8 = cv2.cvtColor((img_rgb_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    score = cv2.quality.QualityBRISQUE_compute(bgr_u8, str(model_path), str(range_path))
    if isinstance(score, tuple) or isinstance(score, list):
        score = score[0]
    if isinstance(score, np.ndarray):
        score = float(score.ravel()[0])
    return float(score)


def compute_noise_proxy(img_rgb_float: np.ndarray) -> float:
    """
    Robust no-reference noise proxy from grayscale MAD:
      sigma ~= 1.4826 * median(|x - median(x)|)
    Lower is typically better for denoising comparison on same image content.
    """
    gray = cv2.cvtColor((img_rgb_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    sigma = 1.4826 * mad
    return float(sigma)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.is_file() and DEFAULT_CKPT.is_file():
        ckpt_path = DEFAULT_CKPT

    model = load_model(ckpt_path)
    img = read_image_rgb_float(input_path)

    inp = to_tensor(img)
    with torch.no_grad():
        out = model(inp)  # DRUNet forward
        out = torch.clamp(out, 0.0, 1.0)

    out_img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Prefer NIQE if local assets already exist. Otherwise fallback to BRISQUE.
    metric_name = "NIQE"
    if NIQE_MODEL_LOCAL.is_file() and NIQE_RANGE_LOCAL.is_file():
        niqe_in = compute_niqe(img, NIQE_MODEL_LOCAL, NIQE_RANGE_LOCAL)
        niqe_out = compute_niqe(out_img, NIQE_MODEL_LOCAL, NIQE_RANGE_LOCAL)
        improvement = niqe_in - niqe_out  # lower is better
    else:
        try:
            metric_name = "BRISQUE"
            brisque_dir = SCRIPT_DIR / ".brisque_assets"
            brisque_model, brisque_range = ensure_brisque_files(brisque_dir)
            niqe_in = compute_brisque(img, brisque_model, brisque_range)
            niqe_out = compute_brisque(out_img, brisque_model, brisque_range)
            improvement = niqe_in - niqe_out  # lower is better
        except Exception:
            metric_name = "NOISE_PROXY_MAD"
            niqe_in = compute_noise_proxy(img)
            niqe_out = compute_noise_proxy(out_img)
            improvement = niqe_in - niqe_out  # lower is better

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(out_path, out_img)

    print(f"Input: {input_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Saved: {out_path}")
    print(f"{metric_name} input:  {niqe_in:.6f}")
    print(f"{metric_name} output: {niqe_out:.6f}")
    print(f"Improvement: {improvement:.6f} (positive is better)")


if __name__ == "__main__":
    main()
