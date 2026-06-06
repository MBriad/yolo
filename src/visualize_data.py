"""Draw YOLO-format ground truth boxes on images."""
from pathlib import Path
import os
import cv2

ROOT = Path(__file__).resolve().parent.parent

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255),
]


def draw_labels(img_dir, label_dir, output_dir, limit=5):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(os.listdir(img_dir))[:limit]

    for fname in images:
        img = cv2.imread(os.path.join(img_dir, fname))
        label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = map(float, parts)
                cls_id = int(cls_id)
                x1 = int((cx - w / 2) * img.shape[1])
                y1 = int((cy - h / 2) * img.shape[0])
                x2 = int((cx + w / 2) * img.shape[1])
                y2 = int((cy + h / 2) * img.shape[0])
                color = COLORS[cls_id % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                name = COCO_NAMES.get(cls_id, str(cls_id))
                cv2.putText(img, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(os.path.join(output_dir, f"gt_{fname}"), img)

    print(f"Saved {len(images)} annotated images to {output_dir}")


if __name__ == "__main__":
    draw_labels(
        str(ROOT / "data" / "coco128" / "images" / "train2017"),
        str(ROOT / "data" / "coco128" / "labels" / "train2017"),
        str(ROOT / "data" / "outputs" / "coco128_viz"),
    )
