if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml")  # Change to your model path if needed
    results = model.train(data="../dataset/config.yaml", epochs=10, batch=8)
