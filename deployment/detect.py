from ultralytics import YOLO

# Load your trained YOLOv11s model
model = YOLO("yolov11s.pt")  

# Run inference on the image and save result in current dir
results = model("testImage.jpg")
results[0].save(filename="testImageResult.jpg")

print("✅ Inference complete. Saved as testImageResult.jpg")
