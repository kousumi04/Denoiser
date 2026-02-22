import os
import glob
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------
# Utility: exposure score
# ---------------------------------------------------------
def exposure_score(img):
    """
    Compute exposure score using mean luminance.
    Range approx [0,1]
    """
    gray = img.convert("L")
    tensor = transforms.ToTensor()(gray)
    return tensor.mean().item()


def _is_image_file(path: str) -> bool:
    return os.path.isfile(path) and Path(path).suffix.lower() in VALID_EXTS


def _list_images(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in files if _is_image_file(p)]


def _resolve_subdir(root: str, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def _pair_by_stem(low_imgs: list[str], high_imgs: list[str]) -> list[tuple[str, str]]:
    low_by_stem = {Path(p).stem: p for p in low_imgs}
    high_by_stem = {Path(p).stem: p for p in high_imgs}
    shared = sorted(set(low_by_stem.keys()) & set(high_by_stem.keys()))
    return [(low_by_stem[s], high_by_stem[s]) for s in shared]


# ---------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------
class LiveNetDataset(Dataset):
    """
    Unified dataset loader for:
        - LOL
        - LOLv2
        - SICE

    Returns:
        input_image  (low exposure)
        target_image (normal exposure)
    """

    def __init__(
        self,
        lol_root=None,
        lolv2_root=None,
        sice_root=None,
        img_size=256,
        exposure_target=0.60,
        exposure_tol=0.05
    ):

        self.pairs = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # -------------------------------------------------
        # LOL DATASET
        # -------------------------------------------------
        if lol_root is not None:
            low_dir = _resolve_subdir(lol_root, ("low", "Low"))
            high_dir = _resolve_subdir(lol_root, ("high", "High", "normal", "Normal"))
            low_imgs = _list_images(low_dir) if low_dir else []
            high_imgs = _list_images(high_dir) if high_dir else []
            self.pairs.extend(_pair_by_stem(low_imgs, high_imgs))

        # -------------------------------------------------
        # LOLv2 DATASET
        # -------------------------------------------------
        if lolv2_root is not None:
            low_dir = _resolve_subdir(lolv2_root, ("Low", "low"))
            high_dir = _resolve_subdir(lolv2_root, ("Normal", "normal", "High", "high"))
            low_imgs = _list_images(low_dir) if low_dir else []
            high_imgs = _list_images(high_dir) if high_dir else []
            self.pairs.extend(_pair_by_stem(low_imgs, high_imgs))

        # -------------------------------------------------
        # SICE DATASET
        # -------------------------------------------------
        if sice_root is not None:

            scene_dirs = [p for p in sorted(glob.glob(os.path.join(sice_root, "*"))) if os.path.isdir(p)]

            for scene in scene_dirs:
                imgs = _list_images(scene)

                if len(imgs) < 2:
                    continue

                scored = []

                for img_path in imgs:
                    try:
                        with Image.open(img_path) as img:
                            score = exposure_score(img.convert("RGB"))
                    except (OSError, ValueError):
                        continue
                    scored.append((img_path, score))

                if len(scored) < 2:
                    continue

                # find target exposure image
                target = min(
                    scored,
                    key=lambda x: abs(x[1] - exposure_target)
                )

                target_path, target_score = target

                # darker images = inputs
                for path, score in scored:

                    if score < target_score - exposure_tol:
                        self.pairs.append((path, target_path))

                    # brighter images ignored automatically

        print(f"[LiveNet Dataset] Total pairs: {len(self.pairs)}")

    # -----------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    # -----------------------------------------------------
    def __getitem__(self, idx):

        inp_path, tgt_path = self.pairs[idx]

        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        inp = self.transform(inp)
        tgt = self.transform(tgt)

        return inp, tgt
