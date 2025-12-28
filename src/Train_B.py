from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A

import segmentation_models_pytorch as smp


@dataclass
class Cfg:
    data_root: Path = Path("data/acre_yolo")
    images_dirname: str = "images"
    masks_dirname: str = "masks"
    splits: Tuple[str, ...] = ("train", "val")

    num_classes: int = 3  # 0=bg,1=crop,2=weed

    # FAST preset
    encoder_name: str = "timm-mobilenetv3_large_100"
    encoder_weights: str = "imagenet"
    model_arch: str = "unet"

    # Training
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    is_gpu: bool = device == "cuda"

    input_size: int = 768
    batch_size: int = 4
    num_workers: int = 4

    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4

    dice_weight: float = 0.5

    # class weight clipping (avoid extreme values if bg dominates)
    class_weight_clip: Tuple[float, float] = (0.2, 5.0)

    out_dir: Path = Path("runs_B_seg/unet_mnv3_fast")
    best_ckpt_name: str = "best.pt"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for mpath in sorted(masks_dir.glob("*.png")):
        ipath = images_dir / f"{mpath.stem}.jpg"
        if ipath.exists():
            pairs.append((ipath, mpath))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No image/mask pairs found in:\n  {images_dir}\n  {masks_dir}")

    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ipath, mpath = self.pairs[idx]
        try:
            img = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        except:
            raise RuntimeError(f"Cannot read image/mask: {ipath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out = self.transform(image=img, mask=mask)
        img = out["image"]
        mask = out["mask"].long()

        return img, mask


def make_transforms(split: str, size: int):
    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(
                0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.Affine(
                translate_percent=(0.0, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            A.pytorch.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            A.Normalize(mean=mean, std=std),
            A.pytorch.ToTensorV2(),
        ])


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = target * num_classes + pred
    cm = torch.bincount(idx.view(-1), minlength=num_classes **
                        2).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def compute_iou_from_cm(cm: torch.Tensor) -> Tuple[List[float], float]:
    ious = []
    for c in range(cm.shape[0]):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp
        denom = tp + fp + fn
        iou = (tp / denom).item() if denom > 0 else float("nan")
        ious.append(iou)

    # mean over classes that are not nan
    valid = [x for x in ious if not (np.isnan(x))]
    miou = float(np.mean(valid))
    return ious, miou


@torch.no_grad()
def compute_dice_from_cm(cm: torch.Tensor) -> List[float]:
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    dice = (2 * tp) / (2 * tp + fp + fn)
    return [d.item() for d in dice]


def estimate_class_weights(train_pairs: List[Tuple[Path, Path]], num_classes: int, clip=(0.2, 5.0)) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)

    for _, mpath in train_pairs:
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        for c in range(num_classes):
            counts[c] += np.sum(mask == c)

    inv = counts.sum() / counts
    inv = inv / inv.mean()  # mean normalization
    inv = np.clip(inv, clip[0], clip[1])

    return torch.tensor(inv, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, scaler, ce_loss, dice_loss, cfg: Cfg) -> float:
    model.train()
    running = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(cfg.device, non_blocking=cfg.is_gpu)
        masks = masks.to(cfg.device, non_blocking=cfg.is_gpu)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(enabled=cfg.is_gpu, device_type=cfg.device):
            out = model(imgs)  # [B,C,H,W]
            loss = ce_loss(out, masks) + cfg.dice_weight * \
                dice_loss(out, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return running / n


@torch.no_grad()
def evaluate(model, loader, cfg: Cfg) -> Dict:
    model.eval()
    cm = torch.zeros((cfg.num_classes, cfg.num_classes),
                     dtype=torch.int64, device=cfg.device)

    for imgs, masks in loader:
        imgs = imgs.to(cfg.device, non_blocking=cfg.is_gpu)
        masks = masks.to(cfg.device, non_blocking=cfg.is_gpu)

        out = model(imgs)
        pred = torch.argmax(out, dim=1)  # [B,H,W]

        for b in range(pred.size(0)):
            cm += confusion_matrix(pred[b], masks[b], cfg.num_classes)

    ious, miou = compute_iou_from_cm(cm)
    dice = compute_dice_from_cm(cm)

    return {"ious": ious, "miou": miou, "dice": dice}


def main():
    CFG = Cfg()

    set_seed(CFG.seed)
    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    # Build pairs
    train_pairs = find_pairs(
        CFG.data_root / CFG.images_dirname / "train",
        CFG.data_root / CFG.masks_dirname / "train",
    )
    val_pairs = find_pairs(
        CFG.data_root / CFG.images_dirname / "val",
        CFG.data_root / CFG.masks_dirname / "val",
    )

    # Data
    train_ds = SegDataset(
        train_pairs, transform=make_transforms("train", CFG.input_size))
    val_ds = SegDataset(
        val_pairs, transform=make_transforms("val", CFG.input_size))

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=CFG.is_gpu
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=CFG.is_gpu
    )

    # Model
    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=CFG.encoder_weights,
        classes=CFG.num_classes,
        activation=None,  # --> logits, [B, C, H, W]
    ).to(CFG.device)

    # Losses
    class_w = estimate_class_weights(
        train_pairs, CFG.num_classes, clip=CFG.class_weight_clip).to(CFG.device)
    ce_loss = nn.CrossEntropyLoss(weight=class_w)

    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    # Optim / sched
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs)

    # AutomaticMixedPrecision
    scaler = torch.amp.GradScaler(enabled=CFG.is_gpu)

    best_miou = -1.0
    best_path = CFG.out_dir / CFG.best_ckpt_name

    # Log file
    log_path = CFG.out_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write(f"device={CFG.device}\n")
        f.write(f"class_weights={class_w.detach().cpu().tolist()}\n\n")

    for epoch in range(1, CFG.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, ce_loss, dice_loss, CFG)
        val_res = evaluate(model, val_loader, CFG)
        scheduler.step()

        ious = val_res["ious"]
        miou = val_res["miou"]
        dice = val_res["dice"]

        msg = (
            f"Epoch {epoch:02d}/{CFG.epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_mIoU={miou:.4f} | "
            f"IoU(bg,crop,weed)={[round(x, 4) for x in ious]} | "
            f"Dice={dice:.4f}"
        )
        print(msg)

        with open(log_path, "a") as f:
            f.write(msg + "\n")

        # Save best
        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_miou": best_miou,
                    "cfg": CFG.__dict__,
                },
                best_path
            )

    print(f"\nDone. Best mIoU={best_miou:.4f} saved to: {best_path}")


if __name__ == "__main__":
    main()
