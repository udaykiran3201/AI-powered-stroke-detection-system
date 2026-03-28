import os
import csv
import numpy as np
from PIL import Image

# Write DIRECTLY to processed folder so preprocess_data is skipped
proc_dir = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed"
classes = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "ischemic"]

for split in ["train", "val"]:
    os.makedirs(os.path.join(proc_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(proc_dir, split, "masks"), exist_ok=True)
    
    csv_path = os.path.join(proc_dir, split, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scan_id"] + classes)
        
        for i in range(16):
            scan_id = f"mock_{split}_{i:03d}"
            
            # Make the PNG directly in the processed dir
            img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            img = Image.fromarray(img_array, mode="L")
            img.save(os.path.join(proc_dir, split, "images", f"{scan_id}.png"))
            
            # Mask
            mask_array = np.zeros((256, 256), dtype=np.uint8)
            mask_array[100:150, 100:150] = 255
            mask = Image.fromarray(mask_array, mode="L")
            mask.save(os.path.join(proc_dir, split, "masks", f"{scan_id}.png"))
            
            labels = [0] * len(classes)
            labels[0] = 1
            writer.writerow([scan_id] + labels)

print("Mock data generated directly into processed folder.")
