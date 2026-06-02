import sys
import time
import cv2
from ultralytics import YOLO

input_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_video.mp4"
output_path = "data/outputs/annotated_video.mp4"

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
t_start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    writer.write(annotated)
    frame_idx += 1
    if frame_idx % 30 == 0:
        elapsed = time.time() - t_start
        print(f"Frame {frame_idx}/{total_frames}, FPS: {frame_idx / elapsed:.1f}")

cap.release()
writer.release()
elapsed = time.time() - t_start
print(f"Done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS)")
print(f"Saved to {output_path}")
