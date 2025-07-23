import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8s model
model = YOLO(r"F:\code_pfe_all\results_models\yolov8s\yolov8s.pt")  # Update path to your model

# Load image
image_path = r"F:\code_pfe_all\results_models\yolov8s\testImage2.jpg"
image = cv2.imread(image_path)
H, W, _ = image.shape  # Original image size

# Run inference
results = model(image)  # Get detection results

# Check if objects were detected
if results[0].boxes.shape[0] == 0:
    print("❌ No objects detected.")
else:
    print(f"🔍 Detected {results[0].boxes.shape[0]} objects!")

# Process detections
for box in results[0].boxes:
    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Convert to int
    confidence = box.conf[0].item()  # Confidence score
    class_id = int(box.cls[0])  # Class ID

    # Get class name from model's class list
    class_name = model.names[class_id]  

    # Print detected objects
    print(f"📌 {class_name}: {confidence:.2f} at ({x_min}, {y_min}, {x_max}, {y_max})")

    # Draw bounding box on image
    color = (0, 255, 0)  # Green
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, f"{class_name} {confidence:.2f}", (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Convert image to BGR for OpenCV display
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Show the image with detections
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
