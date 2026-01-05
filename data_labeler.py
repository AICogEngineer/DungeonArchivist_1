import os
import shutil
import re

# Paths
SOURCE_DIR = "Dungeon_Crawler_Data"
DEST_DIR = "Labeled_Dataset2"

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Allowed image extensions
IMAGE_EXTENSIONS = {".png"}


def to_camel_case(path_parts):
    """
    Converts a list of path parts into camelCase.
    Example: ['Monsters', 'Goblin'] -> 'MonstersGoblin'
    This is done for two reasons:
        1. preserving full filepath because when taking only lowest level directory, items in floors/artefacts 
            and items in weapons/artefacts would be considered same category
        2. camelCase because labels can be easily accessed by using:
            label = filename.split("_", 1)[0]
    """
    words = []
    for part in path_parts:
        split_words = re.split(r"[_\-\s]+", part.lower())
        words.extend(split_words)

    return words[0] + "".join(word.capitalize() for word in words[1:])


for root, _, files in os.walk(SOURCE_DIR):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        # Relative path from SOURCE_DIR
        rel_path = os.path.relpath(root, SOURCE_DIR)

        # Break path into components and convert to camelCase
        path_parts = rel_path.split(os.sep)
        camel_path = to_camel_case(path_parts[1:])

        source_path = os.path.join(root, filename)

        # New filename
        new_filename = f"{camel_path}_{filename}"
        dest_path = os.path.join(DEST_DIR, new_filename)

        shutil.copy2(source_path, dest_path)
        # Use shutil.move(src_path, dest_path) to move instead of copy

print("All images transferred successfully.")