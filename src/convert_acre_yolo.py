import json
import os
from pathlib import Path
import shutil

# === CONFIG ===
PROJECT_ROOT = Path(".")
ACRE_ROOT = PROJECT_ROOT / "data" / "The_ACRE_Crop-Weed_Dataset"
ACRE_DATA = ACRE_ROOT / "data"
COCO_JSON = ACRE_DATA / "ACRE_COCO_annotations.json"
SPLIT_JSON = ACRE_ROOT / "split_dictionary.json"
YOLO_ROOT = PROJECT_ROOT / "data" / "acre_yolo"

# mapping ACRE split --> usual split
SPLIT_MAP = {
    "train": "train",
    "test_dev": "val",
    "test_final": "test",
}

# mapping category_id --> YOLO class_id
CATEGORY_TO_CLASS = {
    1: 0,  # crop
    2: 1,  # weed
}


def main():
    with open(COCO_JSON, "r") as f:
        coco = json.load(f)
    with open(SPLIT_JSON, "r") as f:
        split_dict = json.load(f)

    # from dict "split --> filename" to "filename --> split"
    filename_to_split = {}
    for split_name, files_dict in split_dict.items():
        for fname in files_dict.keys():
            filename_to_split[fname] = split_name

    # dict "image_id --> image_info"
    images_by_id = {img["id"]: img for img in coco["images"]}
    print("Total images:", len(images_by_id))

    # dict "image_id --> image_annotations"
    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    # crete dataset folders
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (YOLO_ROOT / sub).mkdir(parents=True, exist_ok=True)

    num_anns_used = 0
    num_anns_skipped = 0
    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # e.g. "rgb-2022-10-10-15-33-11.jpg"
        base_name = os.path.basename(file_name)
        if base_name not in filename_to_split:
            print(
                f"[WARN] - Unexpected: {base_name} not found in split_dictionary, skipping")
            continue
        # "train"/"val"/"test"
        yolo_split = SPLIT_MAP[filename_to_split[base_name]]

        yolo_lines = []  # class, xc_norm, yc_norm, w_norm, h_norm
        anns = anns_by_image.get(img_id, [])
        for ann in anns:
            category_id = ann["category_id"]
            if category_id not in CATEGORY_TO_CLASS:
                num_anns_skipped += 1
                continue

            # COCO bbox: [x_min, y_min, width, height] in pixel
            x_min, y_min, w, h = ann["bbox"]
            x_c_norm = (x_min + w / 2.0) / width
            y_c_norm = (y_min + h / 2.0) / height
            w_norm = w / width
            h_norm = h / height

            yolo_lines.append(
                f"{CATEGORY_TO_CLASS[category_id]} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

            num_anns_used += 1

        # Dataset creation by copying from original dataset
        src_img_path = ACRE_DATA / file_name
        dst_img_path = YOLO_ROOT / "images" / yolo_split / base_name
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)

        label_path = YOLO_ROOT / "labels" / yolo_split / (Path(base_name).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"Done. Images processed!")
    print(f"Annotations used: {num_anns_used}")
    print(f"Annotations skipped (unknown categories): {num_anns_skipped}")
    print(f"YOLO dataset root: {YOLO_ROOT}")


if __name__ == "__main__":
    main()
