"""
Stroke Detection System — Training Script for Stroke Classifier

Usage:
    python -m app.training.train_classifier \
        --data_dir data/processed \
        --epochs 30 \
        --batch_size 16 \
        --lr 1e-4

This script assumes a directory structure:
    data/processed/
        train/
            images/      ← CT slices (PNG)
            labels.csv   ← scan_id, epidural, intraparenchymal, … , ischemic
        val/
            images/
            labels.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from loguru import logger

from app.models.classifier import StrokeClassifier, STROKE_CLASSES, NUM_CLASSES


# ────────────────────────────────────────
# Dataset
# ────────────────────────────────────────
class StrokeDataset(Dataset):
    """CT scan dataset with multi-label targets."""

    def __init__(self, image_dir: str, labels_csv: str, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(labels_csv)
        self.transform = transform
        self.label_cols = STROKE_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["scan_id"] + ".png")
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Duplicate to 3 channels for pretrained backbone
        image = image.repeat(3, 1, 1)

        labels = torch.tensor(
            [row[col] for col in self.label_cols], dtype=torch.float32
        )
        return image, labels


# ────────────────────────────────────────
# Transforms
# ────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])


# ────────────────────────────────────────
# Training loop
# ────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # Per-class AUC could be computed here with sklearn
    return avg_loss, all_preds, all_labels


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train stroke classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b3")
    parser.add_argument("--output_dir", type=str, default="models/weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Datasets
    train_ds = StrokeDataset(
        os.path.join(args.data_dir, "train", "images"),
        os.path.join(args.data_dir, "train", "labels.csv"),
        transform=train_transforms,
    )
    val_ds = StrokeDataset(
        os.path.join(args.data_dir, "val", "images"),
        os.path.join(args.data_dir, "val", "labels.csv"),
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = StrokeClassifier(backbone_name=args.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # Training
    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "stroke_classifier.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ Best model saved → {save_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
