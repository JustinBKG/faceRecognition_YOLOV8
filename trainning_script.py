from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt") 

# Use the model
model.train(data="./yoloV8_dataset/datasetIndexConfig.yaml", batch=8, imgsz=640, epochs=100)