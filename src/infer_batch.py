from pathlib import Path
import glob
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

image_paths = glob.glob(str(ROOT / "data" / "test_images" / "*.jpg"))
model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
results = model(image_paths)

total = 0
for path, result in zip(image_paths, results):
    n = len(result.boxes)
    total += n
    confs = result.boxes.conf.tolist() if n else []
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"{path}: {n} detections (avg conf {avg_conf:.2f})")

print(f"\nTotal: {total} objects across {len(image_paths)} images")
