from ultralytics import YOLO

# Load models
license_plate_detector = YOLO('../runs/detect/train2/weights/best.pt')

# Run inference with optimized settings
results = license_plate_detector.predict(source="../test_data/car-wbs-MH12FU1014_00000.png", 
                                         save=True )



# Print final detectionss
# print(f"Final license plates detected: {len(filtered_results)}")
