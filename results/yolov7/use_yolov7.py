import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.general import non_max_suppression

# ✅ Load the YOLOv7 model
def load_model(model_path):
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    return model

# ✅ Preprocess the image
def preprocess_image(image_path, img_size=640):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, np.array(image)

# ✅ Decode YOLOv7 outputs correctly
def decode_yolov7_output(output):
    """ Convert YOLOv7 raw output (feature maps) into final bounding boxes. """
    predictions = []

    # YOLOv7 gives a list of 3 feature maps (for different scales)
    for feature_map in output:
        feature_map = feature_map.squeeze(0)  # Remove batch dim (from [1, C, H, W, 18] to [C, H, W, 18])
        feature_map = feature_map.view(-1, feature_map.shape[-1])  # Flatten to (N, 18)

        predictions.append(feature_map)

    # 🔹 Concatenate all scales together (80x80, 40x40, 20x20)
    predictions = torch.cat(predictions, dim=0)

    return predictions

# ✅ Process the decoded predictions
def process_yolo_output(output, conf_threshold=0.25):
    # Step 1: Decode YOLOv7 output
    output = decode_yolov7_output(output)

    # Step 2: Apply NMS (Non-Maximum Suppression)
    predictions = non_max_suppression(output, conf_thres=conf_threshold, iou_thres=0.45)

    if predictions[0] is None:
        print("❌ No objects detected.")
        return []

    detections = predictions[0].cpu().numpy()
    print(f"✅ {len(detections)} object(s) detected.")
    
    return detections

# ✅ Run inference & visualize results
def detect_objects(model, image_path, output_path, conf_threshold=0.25):
    image_tensor, img_cv2 = preprocess_image(image_path)
    img_height, img_width, _ = img_cv2.shape
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        results = model(image_tensor)  # Run inference

    detections = process_yolo_output(results, conf_threshold)

    if len(detections) == 0:
        print("❌ No objects detected.")
    else:
        print(f"✅ {len(detections)} object(s) detected:")
    
    for x1, y1, x2, y2, conf, cls in detections:
        print(f"   🔹 Detected Class {cls} with Confidence: {conf:.2f}")
        cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_cv2, f'Class {cls}: {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_cv2)
    print(f"📸 Output image saved to {output_path}")

# ✅ Run everything
if __name__ == "__main__":
    model_path = r"F:\code_pfe_all\results_models\yolov7\checkpoints1.torchscript.pt"
    image_path = r"F:\code_pfe_all\IP102_DATASET\dataset\prototypes&images&annotations\v1.1\images\IP015\IP015000014.jpg"
    output_path = "output.jpg"

    model = load_model(model_path)
    detect_objects(model, image_path, output_path)
