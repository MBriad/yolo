"""Validate a YOLO-format dataset for common issues."""
import os
import sys
from PIL import Image


def validate(dataset_dir):
    img_dir = os.path.join(dataset_dir, "images", "train2017")
    label_dir = os.path.join(dataset_dir, "labels", "train2017")

    if not os.path.isdir(img_dir):
        print(f"ERROR: image directory not found: {img_dir}")
        return

    images = sorted(os.listdir(img_dir))
    labels = sorted(os.listdir(label_dir))

    img_names = {f.rsplit(".", 1)[0] for f in images}
    label_names = {f.rsplit(".", 1)[0] for f in labels}

    orphan_imgs = img_names - label_names
    orphan_labels = label_names - img_names

    total_boxes = 0
    empty_labels = 0
    invalid_coords = 0
    corrupted = 0

    for fname in images:
        path = os.path.join(img_dir, fname)
        try:
            Image.open(path).verify()
        except Exception:
            corrupted += 1

    for fname in labels:
        path = os.path.join(label_dir, fname)
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            empty_labels += 1
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_coords += 1
                continue
            vals = list(map(float, parts))
            if vals[0] < 0 or not all(0 <= v <= 1 for v in vals[1:]):
                invalid_coords += 1
            total_boxes += 1

    print(f"=== {dataset_dir} Validation Report ===")
    print(f"Images: {len(images)} | Labels: {len(labels)}")
    print(f"Total boxes: {total_boxes}")
    print(f"Orphan images (no label): {len(orphan_imgs)}")
    print(f"Orphan labels (no image): {len(orphan_labels)}")
    print(f"Empty label files: {empty_labels}")
    print(f"Invalid coordinates: {invalid_coords}")
    print(f"Corrupted images: {corrupted}")

    errors = len(orphan_imgs) + len(orphan_labels) + empty_labels + invalid_coords + corrupted
    if errors == 0:
        print("=== PASSED ===")
    else:
        print(f"=== {errors} ERROR(S) FOUND ===")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/coco128"
    validate(path)
