from ultralytics import YOLO
import cv2
import sys
import os

# Dynamically find the ML_ALPR_SCRATCH root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now Python can find sort/ and util.py
from sort.sort import *  # Import from sort/
from src.util import get_car, read_license_plate, write_csv  # Import from src/util.py


results = {}


mot_tracker = Sort()


#load model
coco_model = YOLO('./models/yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')


#load video
cap = cv2.VideoCapture('../test_data/expressed_2103099-uhd_3840_2160_30fps.mp4') 

vehicles = [2, 3, 5, 7]

##Check whether cap.isOpened() is true 
if not cap.isOpened():
    print("Error: Could not open file correctly")
else:
    print("Video file loaded successfully")



#read frames 
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret , frame = cap.read()
    if ret and frame_nmr< 10 :
        results[frame_nmr] = {}
        #detect vehicles 
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        #track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        #detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate


            #assign license plate to car 
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            #crop license plate 
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]


            #process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            cv2.imshow('original_crop', license_plate_crop)
            cv2.imshow('threshold', license_plate_crop_thresh)

            cv2.waitKey(0)
            #read license plate number 

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox':[xcar1, ycar1, xcar2, ycar2]}, 
                                              'license_plate': {'bbox':[x1, y1, x2, y2],
                                                                'text':license_plate_text, 
                                                                'bbox_score': score, 
                                                                'text_score': license_plate_text_score}}

#write results
write_csv(results, './test.csv')





