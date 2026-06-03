from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolov8n.pt")
    results = model.train(
        data="data/custom/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=".",
        name="person_train_v1",
        exist_ok=True,
    )
    print("\nBest model: runs/detect/person_train_v1/weights/best.pt")
