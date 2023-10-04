from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')

if __name__ == '__main__':
# Use the model
    results = model.train(data='GSAI_Images.yaml', epochs=2, imgsz=640)
