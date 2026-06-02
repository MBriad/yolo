import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
