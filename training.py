if __name__ == '__main__':
    from ultralytics import YOLO

    # Load your existing trained model for fine-tuning
    model = YOLO("../models/license_plate_detector.pt")  

    # Start fine-tuning
    results = model.train(
        data="./dataset/config.yaml",  # Dataset configuration
        epochs=50,                      # More epochs for better fine-tuning
        batch=16,                        # Adjust based on GPU
        imgsz=640,                       # Higher resolution for better small object detection
        patience=15,                      # Early stopping if no improvement
        lr0=0.0005,                       # Lower learning rate for fine-tuning
        weight_decay=0.0005,              # Prevents overfitting
        mosaic=0.8,                       # Data augmentation (mild)
        mixup=0.1,                        # Mixup augmentation (lower to avoid confusion)
        hsv_h=0.015,                      # Hue augmentation
        hsv_s=0.7,                        # Saturation augmentation
        hsv_v=0.4,                        # Brightness augmentation
        degrees=5,                        # Small rotation augmentation
        translate=0.1,                    # Small translations
        scale=0.5,                        # Scaling augmentation
        shear=2,                          # Shearing augmentation
        flipud=0.2,                       # Vertical flipping
        fliplr=0.5,                       # Horizontal flipping
        dropout=0.1,                      # Regularization
        close_mosaic=15,                  # Reduces mosaic augmentation after 15 epochs
        cache=True,                        # Cache dataset for faster training
        device='cuda'                      # Use GPU
    )
