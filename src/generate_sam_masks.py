import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= CONFIG =================

PROJECT_ROOT = Path(".")

YOLO_DATA_ROOT = PROJECT_ROOT / "data" / "acre_yolo"
DETECTIONS_ROOT = PROJECT_ROOT / "outputs" / "detections_partA"
MASKS_ROOT = YOLO_DATA_ROOT / "masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_RATIO = 0.10
MASK_TH = 0.6

# YOLO: 0=crop, 1=weed  -> Segm: 0=bg,1=crop,2=weed
YOLO_CLASS_TO_LABEL = {
    0: 1,
    1: 2,
}


# ================= SAM2 UTILS =================
# Since mAP50-95 is not as high as mAP
def expand_box(x1, y1, x2, y2, img_w, img_h, pad_ratio):
    w = x2 - x1
    h = y2 - y1

    pad_w = w * pad_ratio
    pad_h = h * pad_ratio

    x1e = max(0, int(np.floor(x1 - pad_w)))
    y1e = max(0, int(np.floor(y1 - pad_h)))
    x2e = min(img_w - 1, int(np.ceil(x2 + pad_w)))
    y2e = min(img_h - 1, int(np.ceil(y2 + pad_h)))

    return x1e, y1e, x2e, y2e


def generate_masks_for_split(predictor, split: str, detections_json: Path):
    with open(detections_json, "r") as f:
        detections_data = json.load(f)

    out_dir = MASKS_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{split}] Generating SAM2 masks for {len(detections_data)} images...")
    for item in detections_data:
        img_path = PROJECT_ROOT / Path(item["image_path"])

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image {img_path}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        seg_map = np.zeros((h, w), dtype=np.uint8)

        dets = item.get("detections", [])
        if len(dets) == 0:
            mask_path = out_dir / (img_path.stem + ".png")
            cv2.imwrite(str(mask_path), seg_map)
            continue

        dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Set whole image to low inference time
            predictor.set_image(img)

            for det in dets_sorted:
                x1e, y1e, x2e, y2e = expand_box(*det["bbox_xyxy"], w, h, PAD_RATIO)

                patch = img[y1e:y2e, x1e:x2e]
                if patch.size == 0:
                    continue

                if x2e <= x1e or y2e <= y1e:
                    print("UNEXPECTED")
                    continue

                box_patch = np.array([[x1e, y1e, x2e, y2e]], dtype=np.float32)

                # SAM2: set image + predict
                masks, scores, _ = predictor.predict(
                    box=box_patch,
                    multimask_output=False
                )
                best_i = int(np.argmax(scores))
                mask_f = masks[best_i]
                mask = (mask_f > MASK_TH)

                # Update mask area
                mask_crop = mask[y1e:y2e, x1e:x2e]
                submap = seg_map[y1e:y2e, x1e:x2e]
                update_mask = mask_crop & (submap == 0)
                submap[update_mask] = YOLO_CLASS_TO_LABEL[det["cls_id"]]
                seg_map[y1e:y2e, x1e:x2e] = submap

        mask_path = out_dir / (img_path.stem + ".png")
        cv2.imwrite(str(mask_path), seg_map)

    print(f"[{split}] Done. Masks saved to {out_dir}")


def main():
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", device=DEVICE)
    for split in ["train", "val", "test"]:
        detections_json = DETECTIONS_ROOT / f"detections_yolo11n_{split}.json"
        if not detections_json.exists():
            print(f"[{split}] Detections file not found: {detections_json}")
        else:
            generate_masks_for_split(predictor, split, detections_json)


if __name__ == "__main__":
    main()
