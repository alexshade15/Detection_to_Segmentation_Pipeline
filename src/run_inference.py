from ultralytics import YOLO
from pathlib import Path
import json
import torch

from Train_A import img_size, iou, conf

PROJECT_ROOT = Path(".")
YOLO_DATA_ROOT = PROJECT_ROOT / "data" / "acre_yolo"
MODEL_WEIGHTS = PROJECT_ROOT / "runs_A_objdet" / \
    "yolo11n_crop_weed" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "detections_partA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "crop",
    1: "weed",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Loading model from {MODEL_WEIGHTS}")
    model = YOLO(str(MODEL_WEIGHTS))

    for split in ["train", "val", "test"]:
        images_dir = YOLO_DATA_ROOT / "images" / split
        out_json = OUT_DIR / f"detections_yolo11n_{split}.json"

        img_paths = sorted(images_dir.glob("*.jpg"))
        print(f"[{split}] Found {len(img_paths)} images")

        all_results = []
        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                imgsz=img_size,
                conf=conf,
                iou=iou,
                device=DEVICE,
                verbose=False,
            )[0]

            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    detections.append(
                        {
                            "cls_id": int(box.cls[0].item()),
                            "conf": float(box.conf[0].item()),
                            "bbox_xyxy": box.xyxy[0].tolist(),
                        }
                    )

            all_results.append(
                {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "split": split,
                    "detections": detections,
                }
            )

        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"[{split}] Saved detections to {out_json}")


if __name__ == "__main__":
    main()
