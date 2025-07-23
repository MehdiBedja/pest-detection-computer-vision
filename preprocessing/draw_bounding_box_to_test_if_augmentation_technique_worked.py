import cv2
import xml.etree.ElementTree as ET

# Function to parse XML and extract bounding boxes
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image_path, annotation_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Parse the annotation XML to get bounding boxes
    bboxes = parse_xml(annotation_path)
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # Draw a rectangle (bounding box) on the image
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box with thickness of 2
    
    # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", img)
    
    # Save the image with bounding boxes (optional)
    # cv2.imwrite("output_image_with_bboxes.jpg", img)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\images\IP087\augmented_IP087000006_IP087000473.jpg"  # Path to your image file
annotation_path = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\annotations\IP087\augmented_IP087000006_IP087000473.xml"  # Path to your annotation XML file

# Call the function to draw bounding boxes
draw_bounding_boxes(image_path, annotation_path)
