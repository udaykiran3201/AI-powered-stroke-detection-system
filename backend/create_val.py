import os
import csv
import shutil
import random

processed_train_images = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\train\images"
processed_train_csv = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\train\labels.csv"

processed_val_images = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\val\images"
processed_val_csv = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\val\labels.csv"

os.makedirs(processed_val_images, exist_ok=True)

# Read the training labels
with open(processed_train_csv, "r") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = reader[1:]

# Take 100 images randomly for validation
random.shuffle(data)
val_data = data[:100]

with open(processed_val_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in val_data:
        scan_id = row[0]
        src_path = os.path.join(processed_train_images, f"{scan_id}.png")
        dst_path = os.path.join(processed_val_images, f"{scan_id}.png")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        writer.writerow(row)

print("Validation dataset populated!")
