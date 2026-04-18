import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "data_pipeline" / "datasets" / "SIDD_Small_sRGB_Only(Denoising_dataset)" / "Data"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "preprocessed_sidd" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SIDD into paired training patches.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="SIDD Data folder")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Folder to save .pt patches")
    parser.add_argument("--patch-size", type=int, default=128, help="Square crop size")
    parser.add_argument("--patches-per-image", type=int, default=50, help="Number of patches per scene")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-augment", action="store_true", help="Disable flip/rotation augmentation")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def random_crop(noisy: np.ndarray, clean: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = noisy.shape
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Patch size {patch_size} is larger than image size {(h, w)}."
        )

    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    noisy_crop = noisy[top : top + patch_size, left : left + patch_size]
    clean_crop = clean[top : top + patch_size, left : left + patch_size]
    return noisy_crop, clean_crop


def augment(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() > 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)

    if random.random() > 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)

    if random.random() > 0.5:
        k = random.randint(1, 3)
        noisy = np.rot90(noisy, k=k)
        clean = np.rot90(clean, k=k)

    return noisy.copy(), clean.copy()


def find_scene_pair(scene_path: Path) -> tuple[Path | None, Path | None]:
    noisy_path = None
    clean_path = None

    for file in scene_path.iterdir():
        if "NOISY_SRGB" in file.name:
            noisy_path = file
        elif "GT_SRGB" in file.name:
            clean_path = file

    return noisy_path, clean_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        raise FileNotFoundError(f"SIDD input folder not found: {input_root}")

    scenes = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not scenes:
        raise RuntimeError(f"No scene folders found in: {input_root}")

    sample_index = 0
    skipped_scenes = []
    augment_enabled = not args.no_augment

    for scene_path in tqdm(scenes, desc="Scenes"):
        noisy_path, clean_path = find_scene_pair(scene_path)
        if noisy_path is None or clean_path is None:
            skipped_scenes.append(scene_path.name)
            continue

        noisy_img = read_image(noisy_path)
        clean_img = read_image(clean_path)

        if noisy_img.shape != clean_img.shape:
            raise ValueError(
                f"Shape mismatch in {scene_path.name}: noisy={noisy_img.shape}, clean={clean_img.shape}"
            )

        for _ in range(args.patches_per_image):
            noisy_crop, clean_crop = random_crop(noisy_img, clean_img, args.patch_size)

            if augment_enabled:
                noisy_crop, clean_crop = augment(noisy_crop, clean_crop)

            noisy_tensor = torch.from_numpy(noisy_crop).permute(2, 0, 1)
            clean_tensor = torch.from_numpy(clean_crop).permute(2, 0, 1)

            save_dict = {"noisy": noisy_tensor, "clean": clean_tensor}
            save_path = output_root / f"sample_{sample_index:06d}.pt"
            torch.save(save_dict, save_path)
            sample_index += 1

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patches per image: {args.patches_per_image}")
    print(f"Augment: {augment_enabled}")
    print(f"Total patches saved: {sample_index}")
    print(f"Skipped incomplete scenes: {len(skipped_scenes)}")
    if skipped_scenes:
        print("Incomplete scenes:")
        for scene_name in skipped_scenes:
            print(f"  - {scene_name}")


if __name__ == "__main__":
    main()
