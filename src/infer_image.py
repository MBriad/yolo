from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
results = model('data/bus.jpg')

print(f"Detected {len(results[0].boxes)} objects:\n")
for box in results[0].boxes:
    class_id = int(box.cls.item())
    class_name = results[0].names[class_id]
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"  {class_name}: {confidence:.2f} @ [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

import cv2, os
annotated = results[0].plot()
os.makedirs("data/outputs", exist_ok=True)
cv2.imwrite("data/outputs/annotated.jpg", annotated)
print("\nSaved to data/outputs/annotated.jpg")
