import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_restormer.pth")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "inference", "outputs")


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from training.models.restormer import Restormer
except ModuleNotFoundError:
    from models.restormer import Restormer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Restormer denoising on a PNG image.")
    parser.add_argument("input", nargs="?", help="Path to input PNG image.")
    parser.add_argument("--input", dest="input_flag", default=None, help="Path to input PNG image.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (.pth). Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Default: inference/outputs/<input_name>_denoised.png",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "force", "skip"],
        help="auto: metric-based routing, force: always denoise, skip: bypass model.",
    )
    parser.add_argument("--t1", type=float, default=0.01, help="Noise threshold for skip/mild boundary.")
    parser.add_argument("--t2", type=float, default=0.03, help="Noise threshold for mild/strong boundary.")
    parser.add_argument("--t-noise", dest="t_noise", type=float, default=0.0, help="If noise <= t-noise, skip denoising.")
    parser.add_argument("--t-blur", dest="t_blur", type=float, default=0.001, help="Blur threshold on Laplacian variance.")
    parser.add_argument("--t-res", dest="t_res", type=float, default=0.005, help="Residual threshold to skip tiny changes.")
    parser.add_argument("--k", type=float, default=0.02, help="Adaptive blending factor in alpha=sigma/(sigma+k).")
    parser.add_argument("--mild-alpha", dest="mild_alpha", type=float, default=0.35, help="Blend alpha for mild denoising.")
    parser.add_argument("--strong-alpha", dest="strong_alpha", type=float, default=0.85, help="Blend alpha for strong denoising.")
    args = parser.parse_args()
    args.input = args.input_flag or args.input
    if not args.input:
        parser.error("input image is required. Use: python inference_restormer <image.png>")
    return args


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key[len("module."):]] = value
        else:
            cleaned_state[key] = value

    model.load_state_dict(cleaned_state, strict=True)


def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not image_path.lower().endswith(".png"):
        raise ValueError("Input must be a .png file.")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    return image_np


def image_to_tensor(image_np, device):
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = np.clip(tensor, 0.0, 1.0)
    image = Image.fromarray((tensor * 255.0).astype(np.uint8))
    image.save(output_path)


def default_output_path(input_path):
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    name, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{name}_denoised.png")


def grayscale(image_np):
    return 0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2]


def convolve3(gray, kernel):
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def laplacian_variance(gray):
    lap_kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    lap = convolve3(gray, lap_kernel)
    return float(np.var(lap))


def mad_sigma(gray):
    med = float(np.median(gray))
    return float(1.4826 * np.median(np.abs(gray - med)))


def global_std(gray):
    return float(np.std(gray))


def tenengrad(gray):
    sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    gx = convolve3(gray, sobel_x)
    gy = convolve3(gray, sobel_y)
    return float(np.mean(gx * gx + gy * gy))


def compute_metrics(image_np):
    gray = grayscale(image_np)
    lap_var = laplacian_variance(gray)
    return {
        "noise_mad": mad_sigma(gray),
        "noise_std": global_std(gray),
        "noise_lap_var": lap_var,
        "blur_lap_var": lap_var,
        "blur_tenengrad": tenengrad(gray),
    }


def adaptive_alpha(sigma, k):
    if k <= 0:
        return 1.0
    return float(np.clip(sigma / (sigma + k), 0.0, 1.0))


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model = Restormer().to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    input_np = load_image(args.input)
    input_tensor = image_to_tensor(input_np, args.device)
    metrics = compute_metrics(input_np)

    noise_score = metrics["noise_mad"]
    blur_score = metrics["blur_lap_var"]
    base_alpha = adaptive_alpha(noise_score, args.k)
    action = "run"
    strength = "strong"
    alpha = base_alpha

    if args.mode == "skip":
        action = "skip"
    elif args.mode == "force":
        action = "run"
        strength = "strong"
        alpha = 1.0
    else:
        if noise_score <= args.t_noise or noise_score < args.t1:
            action = "skip"
        elif noise_score < args.t2:
            strength = "mild"
            alpha = min(max(base_alpha, args.mild_alpha), args.strong_alpha)
        else:
            strength = "strong"
            alpha = max(base_alpha, args.strong_alpha)

    if blur_score < args.t_blur:
        print("Blur detected (low Laplacian variance). No deblurrer is configured, applying denoiser only.")

    if action == "skip":
        output_tensor = input_tensor
        residual = 0.0
    else:
        with torch.no_grad():
            model_output = model(input_tensor)
        residual = float(torch.mean(torch.abs(model_output - input_tensor)).item())
        if args.mode != "force" and residual < args.t_res:
            output_tensor = input_tensor
            action = "skip_residual"
        else:
            output_tensor = alpha * model_output + (1.0 - alpha) * input_tensor

    output_path = args.output if args.output else default_output_path(args.input)
    tensor_to_image(output_tensor, output_path)
    print(
        "Metrics: "
        f"noise_mad={metrics['noise_mad']:.6f}, "
        f"noise_std={metrics['noise_std']:.6f}, "
        f"lap_var={metrics['noise_lap_var']:.6f}, "
        f"tenengrad={metrics['blur_tenengrad']:.6f}"
    )
    print(
        "Decision: "
        f"mode={args.mode}, action={action}, strength={strength}, "
        f"alpha={alpha:.4f}, residual={residual:.6f}"
    )
    print(f"Denoised image saved to: {output_path}")


if __name__ == "__main__":
    main()
