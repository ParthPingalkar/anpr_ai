from ultralytics import YOLO
import cv2 



#Load the model 
model = YOLO("runs/detect/train7/weights/best.pt") 

#Run inference on image 
results = model("test_data/2103099-uhd_3840_2160_30fps.mp4",save=True, show=True , stream=True, project="src/output_frames")

for frame in results:
    frame.show()
    frame.save(filename="output_frame.jpg")


print("Inference completed!")   
    