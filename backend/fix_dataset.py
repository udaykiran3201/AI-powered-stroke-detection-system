import os
import csv
import shutil

# Make sure this script traverses ALL subdirectories 
# and finds every real image you put in there!

raw_base = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\raw\train\images\Data"
processed_images = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\train\images"
processed_csv = r"c:\Users\ADMIN\Desktop\project of yanthraa\trail 1\task1\backend\data\processed\train\labels.csv"

# Make sure processed directory exists
os.makedirs(processed_images, exist_ok=True)

classes = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "ischemic"]

count = 0
with open(processed_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scan_id"] + classes)
    
    # NORMAL Images
    normal_dir = os.path.join(raw_base, "NORMAL")
    if os.path.exists(normal_dir):
        labels = [0] * len(classes)
        # Walk recursively through all subdirectories
        for root, _, files in os.walk(normal_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    scan_id = f"normal_{count:04d}"
                    src = os.path.join(root, filename)
                    dst = os.path.join(processed_images, f"{scan_id}.png")
                    shutil.copy2(src, dst)
                    writer.writerow([scan_id] + labels)
                    count += 1
                
    # HEMORRHAGIC Images
    hemorrhagic_dir = os.path.join(raw_base, "Hemorrhagic")
    if os.path.exists(hemorrhagic_dir):
        labels = [0] * len(classes)
        labels[1] = 1 # Mark as intraparenchymal hemorrhage
        # Walk recursively through all subdirectories
        for root, _, files in os.walk(hemorrhagic_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    scan_id = f"hemorrhagic_{count:04d}"
                    src = os.path.join(root, filename)
                    dst = os.path.join(processed_images, f"{scan_id}.png")
                    shutil.copy2(src, dst)
                    writer.writerow([scan_id] + labels)
                    count += 1

print(f"Dataset flattened! {count} total scans processed successfully.")
