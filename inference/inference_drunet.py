import argparse
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.drunet import DRUNet  # noqa: E402


DEFAULT_CKPT_SIDD = PROJECT_ROOT / "checkpoints" / "best_drunet_finetuned_sidd.pth"
DEFAULT_CKPT_DIV2K = PROJECT_ROOT / "checkpoints" / "best_drunet_div2k.pth"
DEFAULT_CKPT_UNIFIED_BEST = PROJECT_ROOT / "checkpoints" / "best_drunet_unified.pt"
DEFAULT_CKPT_UNIFIED_FINAL = PROJECT_ROOT / "checkpoints" / "drunet_unified_final.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "inference" / "outputs"
COMMON_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".jfif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRUNet inference on local or URL image.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to local image (any extension readable by PIL, e.g. png/jpg/webp/tiff/bmp/gif).",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to local image.")
    parser.add_argument("--url", type=str, default=None, help="Image URL from internet.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt/.pth)")
    parser.add_argument("--output", type=str, default=None, help="Path to save denoised image.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Manual noise prior in [0,1]. If omitted, auto-estimate from image.",
    )
    return parser.parse_args()


def choose_checkpoint(user_ckpt: str | None) -> Path:
    if user_ckpt:
        ckpt = Path(user_ckpt)
    elif DEFAULT_CKPT_UNIFIED_BEST.is_file():
        ckpt = DEFAULT_CKPT_UNIFIED_BEST
    elif DEFAULT_CKPT_UNIFIED_FINAL.is_file():
        ckpt = DEFAULT_CKPT_UNIFIED_FINAL
    elif DEFAULT_CKPT_SIDD.is_file():
        ckpt = DEFAULT_CKPT_SIDD
    else:
        ckpt = DEFAULT_CKPT_DIV2K

    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Train first or pass --checkpoint explicitly."
        )
    return ckpt


def load_checkpoint(model: DRUNet, ckpt_path: Path, device: str) -> None:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

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


def download_image(url: str) -> Path:
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        tmp_path = Path(f.name)
    urllib.request.urlretrieve(url, str(tmp_path))
    return tmp_path


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    if suffix and suffix not in COMMON_IMAGE_EXTS:
        # Keep permissive behavior: still try to open with PIL in case plugin/format is available.
        # This allows additional formats like HEIC if PIL plugins are installed.
        pass
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Unsupported or corrupted image file: {path}. "
            "Use a standard image extension or install PIL plugin support for your format."
        ) from exc
    return np.asarray(img, dtype=np.float32) / 255.0


def tensor_from_np(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def save_image(img_tensor: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(out_path)


def estimate_sigma(img: np.ndarray) -> float:
    """
    Fast robust estimator:
      gray = 0.2989 R + 0.5870 G + 0.1140 B
      sigma ~ 1.4826 * median(|gray - median(gray)|)
    Clipped to [0.01, 0.25] to avoid unstable extremes.
    """
    gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    sigma = 1.4826 * mad
    return float(np.clip(sigma, 0.01, 0.25))


def default_output_path(input_name: str) -> Path:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(input_name).stem
    return DEFAULT_OUT_DIR / f"{stem}_drunet_denoised.png"


def main() -> None:
    args = parse_args()
    input_arg = args.input if args.input is not None else args.input_path
    if not input_arg and not args.url:
        raise ValueError("Provide --input <path> or --url <image_url>")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    ckpt = choose_checkpoint(args.checkpoint)
    model = DRUNet().to(args.device)
    load_checkpoint(model, ckpt, args.device)
    model.eval()

    downloaded_tmp = None
    if args.url:
        downloaded_tmp = download_image(args.url)
        img_path = downloaded_tmp
        input_name = Path(urllib.parse.urlparse(args.url).path).name or "url_image"
    else:
        img_path = Path(input_arg)
        input_name = img_path.name

    img_np = read_image(img_path)
    inp = tensor_from_np(img_np, args.device)

    sigma_value = args.sigma if args.sigma is not None else estimate_sigma(img_np)
    sigma_value = float(np.clip(sigma_value, 0.0, 1.0))
    sigma = torch.tensor([[sigma_value]], dtype=inp.dtype, device=inp.device)

    with torch.no_grad():
        out = model(inp, sigma=sigma)
        out = torch.clamp(out, 0.0, 1.0)

    out_path = Path(args.output) if args.output else default_output_path(input_name)
    save_image(out, out_path)

    if downloaded_tmp is not None and downloaded_tmp.exists():
        try:
            downloaded_tmp.unlink()
        except OSError:
            pass

    print(f"Checkpoint: {ckpt}")
    print(f"Sigma used: {sigma_value:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
