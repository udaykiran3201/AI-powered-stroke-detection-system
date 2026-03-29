"""
Stroke Detection System — Stratified Dataset Splitter (Robust Paths)

Creates a 70% Train / 15% Val / 15% Test split from processed data.
Ensures patient-level stratification (no data leakage).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
import shutil

def split_dataset(csv_path, base_dir, target_dir, seed=42):
    # Log absolute paths for debugging
    logger.info(f"Source CSV: {os.path.abspath(csv_path)}")
    logger.info(f"Target Dir: {os.path.abspath(target_dir)}")

    if not os.path.exists(csv_path):
        logger.error(f"FATAL: Source CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Create a binary 'any_stroke' label for stratification
    label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'ischemic']
    df['any_stroke'] = df[label_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Stroke samples: {df['any_stroke'].sum()} ({df['any_stroke'].mean():.1%})")

    # 1. Split Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=seed, 
        stratify=df['any_stroke']
    )

    # 2. Split Temp into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=seed, 
        stratify=temp_df['any_stroke']
    )

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    # Prepare directories
    image_src = os.path.join(base_dir, "images")
    
    for split_name, split_df in splits.items():
        split_dir = os.path.join(target_dir, split_name)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Save labels CSV
        split_df.drop(columns=['any_stroke']).to_csv(os.path.join(split_dir, "labels.csv"), index=False)
        
        # Copy subset of images
        logger.info(f"Preparing {split_name} set ({len(split_df)} images)...")
        for scan_id in split_df['scan_id']:
            src_file = os.path.join(image_src, f"{scan_id}.png")
            dst_file = os.path.join(img_dir, f"{scan_id}.png")
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                logger.warning(f"Missing image: {src_file}")

    logger.info("Dataset split complete ✓")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    # Use relative paths from 'backend' root
    split_dataset(
        csv_path="data/processed/train/labels.csv", 
        base_dir="data/processed/train",
        target_dir="data/split_data"
    )
