from ultralytics import YOLO

img_size = 1024
conf = 0.25
iou = 0.7


def train_partA():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="configs/acre_crop_weed.yaml",
        epochs=50,
        imgsz=img_size,
        batch=16,
        patience=10,
        project="runs_A_objdet",
        name="yolo11n_crop_weed",
        verbose=True,
        conf=conf,
        iou=iou
    )

    return results


if __name__ == "__main__":
    train_partA()
