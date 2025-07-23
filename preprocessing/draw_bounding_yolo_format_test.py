import cv2

# Function to draw bounding boxes from YOLO annotations
def draw_bboxes(image_path, yolo_annotation_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Read YOLO annotations (one line per object)
    with open(yolo_annotation_path, 'r') as f:
        annotations = f.readlines()

    # Get image dimensions
    height, width, _ = image.shape
    
    # Draw bounding boxes for each annotation
    for annotation in annotations:
        # Split annotation into components
        class_id, x_center, y_center, box_width, box_height = map(float, annotation.strip().split())

        # Convert YOLO format (normalized) to pixel values
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the rectangle (bounding box) on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness of 2

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r"F:\code_pfe_all\IP102_DATASET\dataset\final_models\yolo_dataset\dataset_Yolo_version_13pestTypes\test\images\IP016000479.jpg"  # Path to the image
yolo_annotation_path = r"F:\code_pfe_all\IP102_DATASET\dataset\final_models\yolo_dataset\dataset_Yolo_version_13pestTypes\test\labels\IP016000479.txt"  # Path to the YOLO annotation file

draw_bboxes(image_path, yolo_annotation_path)
