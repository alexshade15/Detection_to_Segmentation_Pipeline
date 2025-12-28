import random
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from Train_B import (
    Cfg,
    make_transforms,
    confusion_matrix,
    compute_iou_from_cm,
    compute_dice_from_cm,
)

from eval_SAM2_masks import colorize_mask


def tensor_to_uint8_rgb(x_chw: torch.Tensor, mean, std) -> np.ndarray:
    x = x_chw.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))  # HWC

    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    x = (x * std + mean) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def main():
    CFG = Cfg()
    split = "test"
    samples = 10
    random.seed(CFG.seed)

    ckpt_path = CFG.out_dir / CFG.best_ckpt_name
    images_dir = CFG.data_root / CFG.images_dirname / split
    masks_dir = CFG.data_root / CFG.masks_dirname / split
    out_dir = Path("outputs/qualitative_UNet_masks") / split
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() == ".jpg"]
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    if samples < len(imgs):
        visual_imgs = random.sample(imgs, samples)
    else:
        visual_imgs = imgs

    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,
        classes=CFG.num_classes,
        activation=None,
    )
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(CFG.device)
    model.eval()

    tfm = make_transforms("eval", CFG.input_size)
    cm = torch.zeros((CFG.num_classes, CFG.num_classes),
                     dtype=torch.int64, device=CFG.device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    with torch.inference_mode():
        for img_path in imgs:
            gt_path = masks_dir / f"{img_path.stem}.png"
            if not gt_path.exists():
                print(f"[WARN] Missing GT mask for {gt_path.name}, skipping")
                continue

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[WARN] Could not read {img_path}, skipping")
                continue
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is None:
                print(f"[WARN] Could not read {gt_path}, skipping")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gt = gt.astype(np.int64)

            # apply same transforms to image and mask
            out = tfm(image=img_rgb, mask=gt)
            x = out["image"].unsqueeze(0).to(CFG.device)  # (1,3,H,W)
            gt_t = out["mask"].to(CFG.device).long()  # (H,W)

            logits = model(x)               # (1,C,H,W)
            pred = torch.argmax(logits, dim=1).squeeze(0)

            # metrics update
            cm += confusion_matrix(pred, gt_t, CFG.num_classes)

            if img_path in visual_imgs:
                vis_img = tensor_to_uint8_rgb(out["image"], mean, std)

                gt_np = gt_t.detach().cpu().numpy().astype(np.uint8)
                pr_np = pred.detach().cpu().numpy().astype(np.uint8)
                gt_rgb = colorize_mask(gt_np)
                pr_rgb = colorize_mask(pr_np)

                left = cv2.addWeighted(vis_img, 1.0, gt_rgb, 0.45, 0.0)
                right = cv2.addWeighted(vis_img, 1.0, pr_rgb, 0.45, 0.0)

                panel = np.concatenate([left, right], axis=1)
                panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

                cv2.imwrite(
                    str(out_dir / f"{img_path.stem}_GTvsPRED.jpg"), panel_bgr)

    ious, miou = compute_iou_from_cm(cm)
    dice = compute_dice_from_cm(cm)

    # report
    class_names = ["bg", "crop", "weed"][:CFG.num_classes]
    print("\n=== TEST Metrics ===")
    for i, name in enumerate(class_names):
        print(f"{name:>5s} | IoU: {ious[i]:.4f} | Dice: {dice[i]:.4f}")

    print(f"\nmIoU(all classes) = {miou:.4f}")

    # Foreground metrics
    fg = [1, 2]
    print(f"mIoU(fg=crop+weed) = {float(np.mean([ious[i] for i in fg])):.4f}")
    print(f"mean Dice(fg)     = {float(np.mean([dice[i] for i in fg])):.4f}")

    print(f"\nQualitative panels saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
