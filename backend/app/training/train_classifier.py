"""
Stroke Detection — Professional Training Pipeline
Optimised for: High-accuracy stroke classification on CPU (16GB RAM)
Backbone: EfficientNet-B0 (Low memory, high precision)
Stratification: 70/15/15 Train/Val/Test
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from app.models.classifier import StrokeClassifier, STROKE_CLASSES, NUM_CLASSES
import timm

# ───── Config ─────

class StrokeDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.split_dir = split_dir
        self.image_dir = os.path.join(split_dir, "images")
        self.df = pd.read_csv(os.path.join(split_dir, "labels.csv"))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["scan_id"] + ".png")
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        # 3 channels for backbone
        image = image.repeat(3, 1, 1)
        labels = torch.tensor([row[col] for col in STROKE_CLASSES], dtype=torch.float32)
        return image, labels

def create_model(backbone='efficientnet_b0', num_classes=6):
    model = StrokeClassifier(backbone_name=backbone, num_classes=num_classes, pretrained=True)
    return model

def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Binary 'Is Stroke' vs 'Normal'
    y_true_any = (y_true.sum(axis=1) > 0).astype(int)
    y_pred_any = (y_pred_binary.sum(axis=1) > 0).astype(int)
    
    acc = accuracy_score(y_true_any, y_pred_any)
    f1 = f1_score(y_true_any, y_pred_any, zero_division=0)
    rec = recall_score(y_true_any, y_pred_any, zero_division=0)
    prec = precision_score(y_true_any, y_pred_any, zero_division=0)
    
    return {"acc": acc, "f1": f1, "recall": rec, "precision": prec}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Progress indicator every 10 batches
        if i % 10 == 0:
            logger.info(f"  Batch [{i}/{len(loader)}]... (Crunching images)")
            
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/split_data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Transforms (Stronger Augmentation)
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    # Load Splits
    train_ds = StrokeDataset(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds = StrokeDataset(os.path.join(args.data_dir, "val"), transform=val_tf)
    
    # 16GB RAM Laptop: Keep num_workers low on Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Loss with pos_weight (Balance stroke classes)
    # Calculate pos_weight based on train dataset
    labels = train_ds.df[STROKE_CLASSES].values
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    # SAFETY: If a class has 0 samples, don't use Billions. Cap weights.
    # We use a base weight of 1.0 and increase if needed, maxing at 10.0
    weights_np = np.ones(NUM_CLASSES)
    for i, count in enumerate(pos_counts):
        if count > 0:
            weights_np[i] = min(neg_counts[i] / (count + 1e-6), 10.0)
    
    pos_weight = torch.tensor(weights_np, dtype=torch.float32).to(device)
    logger.info(f"Stable class weights (pos_weight): {pos_weight.cpu().numpy()}")

    model = create_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    best_f1 = 0
    patience = 5
    no_improve = 0

    os.makedirs("models/weights", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics, y_true, y_pred = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        duration = time.time() - t0
        logger.info(f"Epoch [{epoch}/{args.epochs}] — Loss: {tr_loss:.4f}/{val_loss:.4f} | Acc: {metrics['acc']:.1%} | F1: {metrics['f1']:.1%} | Rec: {metrics['recall']:.1%} | Time: {duration:.1f}s")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "models/weights/stroke_classifier.pth")
            logger.info("✓ Better F1 reached! Model saved to models/weights/stroke_classifier.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered. Finished.")
                break

    # Final Test evaluation
    logger.info("Running final evaluation on Test set...")
    test_ds = StrokeDataset(os.path.join(args.data_dir, "test"), transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    model.load_state_dict(torch.load("models/weights/stroke_classifier.pth", weights_only=True))
    _, test_metrics, y_t, y_p = validate(model, test_loader, criterion, device)
    
    logger.info("=== FINAL TEST METRICS ===")
    logger.info(f"Accuracy:  {test_metrics['acc']:.2%}")
    logger.info(f"F1-Score:  {test_metrics['f1']:.2%}")
    logger.info(f"Recall:    {test_metrics['recall']:.2%}")
    logger.info(f"Precision: {test_metrics['precision']:.2%}")
    
    # Detailed report for stroke subtypes
    report = classification_report(y_t, (y_p > 0.45).astype(int), target_names=STROKE_CLASSES)
    print("\nDetailed Per-Class Performance:")
    print(report)

if __name__ == "__main__":
    main()
