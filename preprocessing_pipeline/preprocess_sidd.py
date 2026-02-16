import os
import random
import torch
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
INPUT_ROOT = "C:/Users/kousu/Desktop/SanDisk Hackathon/AIPE/datasets/SIDD_Small_sRGB_Only(Denoising_dataset)/Data"
OUTPUT_ROOT = "C:/Users/kousu/Desktop/SanDisk Hackathon/AIPE/datasets/preprocessed_sidd/train"
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 50   # Number of patches per scene
AUGMENT = True

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def random_crop(noisy, clean, patch_size):
    h, w, _ = noisy.shape
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    noisy_crop = noisy[top:top+patch_size, left:left+patch_size]
    clean_crop = clean[top:top+patch_size, left:left+patch_size]

    return noisy_crop, clean_crop


def augment(noisy, clean):
    if random.random() > 0.5:
        noisy = np.flip(noisy, axis=1)
        clean = np.flip(clean, axis=1)

    if random.random() > 0.5:
        noisy = np.flip(noisy, axis=0)
        clean = np.flip(clean, axis=0)

    if random.random() > 0.5:
        noisy = np.rot90(noisy)
        clean = np.rot90(clean)

    return noisy.copy(), clean.copy()


def main():
    sample_index = 0

    scenes = os.listdir(INPUT_ROOT)

    for scene in tqdm(scenes):
        scene_path = os.path.join(INPUT_ROOT, scene)

        if not os.path.isdir(scene_path):
            continue

        noisy_path = None
        clean_path = None

        for file in os.listdir(scene_path):
            if "NOISY_SRGB" in file:
                noisy_path = os.path.join(scene_path, file)
            elif "GT_SRGB" in file:
                clean_path = os.path.join(scene_path, file)

        if noisy_path is None or clean_path is None:
            continue

        noisy_img = read_image(noisy_path)
        clean_img = read_image(clean_path)

        for _ in range(PATCHES_PER_IMAGE):
            noisy_crop, clean_crop = random_crop(noisy_img, clean_img, PATCH_SIZE)

            if AUGMENT:
                noisy_crop, clean_crop = augment(noisy_crop, clean_crop)

            noisy_tensor = torch.from_numpy(noisy_crop).permute(2, 0, 1)
            clean_tensor = torch.from_numpy(clean_crop).permute(2, 0, 1)

            save_dict = {
                "noisy": noisy_tensor,
                "clean": clean_tensor
            }

            save_path = os.path.join(
                OUTPUT_ROOT, f"sample_{sample_index:06d}.pt"
            )

            torch.save(save_dict, save_path)
            sample_index += 1

    print(f"\nTotal patches saved: {sample_index}")


if __name__ == "__main__":
    main()
