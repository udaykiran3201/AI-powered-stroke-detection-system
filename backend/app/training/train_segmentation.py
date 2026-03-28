"""
Stroke Detection System — Training Script for U-Net Segmentation

Usage:
    python -m app.training.train_segmentation \
        --data_dir data/processed \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-3

Directory structure:
    data/processed/
        train/
            images/   ← CT slices (PNG, grayscale)
            masks/    ← Binary masks (PNG, same filename)
        val/
            images/
            masks/
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from loguru import logger

from app.models.segmentation import UNet


# ────────────────────────────────────────
# Dataset
# ────────────────────────────────────────
class SegmentationDataset(Dataset):
    """Paired CT image + binary mask dataset."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image = transforms.ToTensor()(image)  # (1, H, W)
        mask = transforms.ToTensor()(mask)    # (1, H, W)
        mask = (mask > 0.5).float()           # binarise

        if self.transform:
            image = self.transform(image)

        return image, mask


# ────────────────────────────────────────
# Dice loss
# ────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + (
            1 - self.bce_weight
        ) * self.dice(logits, targets)


# ────────────────────────────────────────
# Metrics
# ────────────────────────────────────────
def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-8)).item()


# ────────────────────────────────────────
# Training
# ────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_iou += compute_iou(outputs, masks) * images.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_iou / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            total_iou += compute_iou(outputs, masks) * images.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_iou / n


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="models/weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_ds = SegmentationDataset(
        os.path.join(args.data_dir, "train", "images"),
        os.path.join(args.data_dir, "train", "masks"),
    )
    val_ds = SegmentationDataset(
        os.path.join(args.data_dir, "val", "images"),
        os.path.join(args.data_dir, "val", "masks"),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = UNet(in_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_IoU={train_iou:.4f}  val_IoU={val_iou:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "unet_segmentation.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ Best model saved → {save_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
