import os
import shutil

# Paths
SOURCE_DIR = "Dungeon_Crawler_Data"        # root directory with subfolders
DEST_DIR = "Labeled_Dataset"     # destination folder

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Allowed image extensions
IMAGE_EXTENSIONS = {".png"}

for root, _, files in os.walk(SOURCE_DIR):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        # Parent directory name = class label
        class_label = os.path.basename(root)

        src_path = os.path.join(root, filename)

        # New filename to prevent overwriting
        new_filename = f"{class_label}_{filename}"
        dest_path = os.path.join(DEST_DIR, new_filename)

        shutil.copy2(src_path, dest_path)
        # Use shutil.move(...) instead if you want to move files

print("All images transferred successfully.")