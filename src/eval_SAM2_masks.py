import random
from pathlib import Path

import cv2
import numpy as np

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colors = {
        0: (0, 0, 0),         # bg
        1: (255, 0, 0),     # crop
        2: (0, 0, 255),     # weed
    }
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, c in colors.items():
        out[mask == k] = c
    return out

def main():
    samples = 10
    split = "test"

    random.seed(42)

    data_root = Path("data/acre_yolo")
    images_dir = data_root / "images" / split
    masks_dir = data_root / "masks" / split

    out_dir = Path("outputs/qualitative_SAM2_masks") / split
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() == ".jpg"]
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    if samples < len(imgs):
        imgs = random.sample(imgs, samples)

    print(f"[{split}] Saving {len(imgs)} check to: {out_dir}")

    for img_path in imgs:
        stem = img_path.stem
        gt_path = masks_dir / f"{stem}.png"
        out_path = out_dir / f"{stem}.jpg"
        if not gt_path.exists():
            print(
                f"[WARN] Missing mask for {img_path.name} -> {gt_path.name}, skipping")
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read image {img_path}, skipping")
            continue

        mask = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] Could not read mask {gt_path}, skipping")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = colorize_mask(mask)
        over = cv2.addWeighted(img_rgb, 1.0, mask_rgb, .45, 0.0)

        panel = np.concatenate([img_rgb, over, mask_rgb], axis=1)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_path), panel_bgr)

    print("Done.")


if __name__ == "__main__":
    main()
