from ultralytics import YOLO

import cv2

#load model
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')


#load video
cap = cv2.VideoCapture('./test_data/') 




#read frames 

#detect vehicles 

#track vehicles

#detect license plates

#assign license plate to car 

#crop license plate 

#process license plate