from ultralytics import YOLO

#Load a model 
model = YOLO("yolov8n.yaml")

#Use the model
results = model.train(data="../dataset/config.yaml", epochs=1) #Train the model


