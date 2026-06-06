from pathlib import Path
import cv2
import os
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
results = model(str(ROOT / "data" / "bus.jpg"))

print(f"Detected {len(results[0].boxes)} objects:\n")
for box in results[0].boxes:
    class_id = int(box.cls.item())
    class_name = results[0].names[class_id]
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"  {class_name}: {confidence:.2f} @ [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

annotated = results[0].plot()
out_dir = ROOT / "data" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)
print(f"\nSaved to data/outputs/annotated.jpg")
