from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam (camera index 0)")
        exit(1)

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
