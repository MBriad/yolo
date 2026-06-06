"""Validate a YOLO-format dataset for common issues."""
from pathlib import Path
import os
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def validate_split(img_dir, label_dir, name):
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
        try:
            Image.open(os.path.join(img_dir, fname)).verify()
        except Exception:
            corrupted += 1

    for fname in labels:
        with open(os.path.join(label_dir, fname)) as f:
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

    print(f"\n  [{name}] {len(images)} images | {len(labels)} labels | {total_boxes} boxes")
    errors = len(orphan_imgs) + len(orphan_labels) + empty_labels + invalid_coords + corrupted
    if orphan_imgs:
        print(f"    Orphan images: {len(orphan_imgs)}")
    if orphan_labels:
        print(f"    Orphan labels: {len(orphan_labels)}")
    if empty_labels:
        print(f"    Empty labels: {empty_labels}")
    if invalid_coords:
        print(f"    Invalid coords: {invalid_coords}")
    if corrupted:
        print(f"    Corrupted: {corrupted}")
    if errors == 0:
        print(f"    OK")
    return errors


def validate(dataset_dir):
    print(f"=== {dataset_dir} Validation ===")
    total_errors = 0
    found = False

    img_base = os.path.join(dataset_dir, "images")
    lbl_base = os.path.join(dataset_dir, "labels")
    if os.path.isdir(img_base) and os.path.isdir(lbl_base):
        for sub in sorted(os.listdir(img_base)):
            img_dir = os.path.join(img_base, sub)
            lbl_dir = os.path.join(lbl_base, sub)
            if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
                total_errors += validate_split(img_dir, lbl_dir, sub)
                found = True

    if not found:
        print("ERROR: no image/label subdirectories found")
        return

    if total_errors == 0:
        print("\n=== PASSED ===")
    else:
        print(f"\n=== {total_errors} ERROR(S) ===")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "coco128")
    validate(path)
