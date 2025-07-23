from ultralytics import YOLO
import cv2
import os
from collections import Counter

# Load model and image paths
model_path = r"F:\code_pfe_all\results_models\yolov8m\best.pt"
image_path = r"C:\Users\DELL\Downloads\images (10).jpg"
output_path = r"C:\Users\DELL\Downloads\output.jpg"  # Output image

# Ensure paths exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Load YOLO model and image
model = YOLO(model_path)
image = cv2.imread(image_path)

# Run inference
results = model(image , conf=0.5 )

# Initialize a counter for detected classes
class_counts = Counter()

# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        
        # Get class name
        class_name = model.names[class_id]
        print(model.names)  # Check if class names exist
        class_counts[class_name] += 1  # Count occurrences

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display class and confidence
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print detection summary
total_objects = sum(class_counts.values())
print(f"Total objects detected: {total_objects}")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Save the image with detections
cv2.imwrite(output_path, image)
print(f"Detection saved to {output_path}")
