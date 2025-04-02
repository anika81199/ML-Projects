import sys
print("Python interpreter:", sys.executable)

try:
    import sklearn
    print("✅ scikit-learn version:", sklearn.__version__)
except ImportError as e:
    print("❌ scikit-learn not found!", e)
    sys.exit(1)
    

import sys
print("Using Python:", sys.executable)

from pathlib import Path
import shutil
import random
from sklearn.model_selection import train_test_split

# === CONFIG ===
script_dir = Path(__file__).parent.absolute()
BASE_DIR = script_dir / "dataset"

IMAGES_DIR = BASE_DIR / "images"
ANNOTATION_DIR = BASE_DIR / "annotations"  # not used in split
LABELS_DIR = BASE_DIR / "labels"

print(f"Image directory: {IMAGES_DIR}")
print(f"Annotation directory: {ANNOTATION_DIR}")
print(f"Label output directory: {LABELS_DIR}")

# These will be created
SPLIT_DIRS = {
    "train": {
        "images": IMAGES_DIR / "train",
        "labels": LABELS_DIR / "train"
    },
    "val": {
        "images": IMAGES_DIR / "val",
        "labels": LABELS_DIR / "val"
    }
}

# === SETUP ===
def setup_dirs():
    for split in SPLIT_DIRS.values():
        for path in split.values():
            path.mkdir(parents=True, exist_ok=True)

# === GET VALID IMAGES ===
def get_valid_image_paths():
    valid_images = []
    for img_path in IMAGES_DIR.glob("*.[pj][np]g"):  # .jpg or .png
        label_path = LABELS_DIR / img_path.with_suffix(".txt").name
        if label_path.exists():
            valid_images.append(img_path)
    return valid_images

# === SPLIT + SAVE ===
def split_and_save(valid_images, split_ratio=0.2):
    train_imgs, val_imgs = train_test_split(valid_images, test_size=split_ratio, random_state=42)

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in split_imgs:
            label_path = LABELS_DIR / img_path.with_suffix(".txt").name

            dst_img = SPLIT_DIRS[split_name]["images"] / img_path.name
            dst_label = SPLIT_DIRS[split_name]["labels"] / label_path.name

            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)

    print(f"✅ Split done: {len(train_imgs)} train, {len(val_imgs)} val")

# === MAIN ===
if __name__ == "__main__":
    setup_dirs()
    valid_images = get_valid_image_paths()
    split_and_save(valid_images)

