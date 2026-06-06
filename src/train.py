from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    results = model.train(
        data=str(ROOT / "data" / "custom" / "dataset.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(ROOT),
        name="person_train_v1",
        exist_ok=True,
    )
    print(f"\nBest model: {ROOT / 'runs' / 'detect' / 'person_train_v1' / 'weights' / 'best.pt'}")
