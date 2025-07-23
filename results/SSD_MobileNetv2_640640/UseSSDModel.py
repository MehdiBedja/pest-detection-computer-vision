import tensorflow as tf
import numpy as np
import cv2
import os

# Path to your exported saved_model directory
SAVED_MODEL_PATH = r"F:\code_pfe_all\results_models\SSD_MobileNetv2_640640\exported_model\saved_model"

# Load the model
print('Loading model...', end='')
detection_model = tf.saved_model.load(SAVED_MODEL_PATH)
predict_fn = detection_model.signatures['serving_default']
print('Done!')

# Load an image (change path as needed)
IMAGE_PATH = r"C:\Users\DELL\Downloads\Hemiptera-Auchenorrhyncha-Cicadellidae-Idiocerus-Leafhoppers-C.jpg"
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_np = np.expand_dims(image_rgb, axis=0)

# Convert image to tensor
input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)

# Perform detection
outputs = predict_fn(input_tensor=input_tensor)

# Extract detection results
boxes = outputs['detection_boxes'].numpy()[0]
scores = outputs['detection_scores'].numpy()[0]
classes = outputs['detection_classes'].numpy()[0].astype(int)

# Draw boxes on the image
HEIGHT, WIDTH, _ = image.shape
for i in range(len(scores)):
    if scores[i] > 0.5:  # Threshold to visualize detections
        box = boxes[i] * np.array([HEIGHT, WIDTH, HEIGHT, WIDTH])
        y_min, x_min, y_max, x_max = box.astype(int)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f'Class {classes[i]}: {scores[i]:.2f}'
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save or display the result
cv2.imwrite('detected_image.jpg', image)
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
