import os
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import random
import numpy as np

# Function to parse XML and extract bounding boxes
def parse_xml(xml_path):
    try:
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
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return []

# Function to calculate IoU (Intersection over Union)
def calculate_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

# Function to apply photometric transformations
def apply_photometric_transformations(image):
    try:
        # Randomly adjust brightness
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Randomly adjust contrast
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
    except Exception as e:
        print(f"Error applying photometric transformations: {e}")
    return image

# Function to paste objects with overlap checking
def paste_object_on_image_with_overlap_check(source_img, target_img, source_bboxes, target_bboxes, iou_threshold=0.1):
    new_bboxes = []
    pasted = False  # Flag to check if at least one object is successfully pasted

    for bbox in source_bboxes:
        xmin, ymin, xmax, ymax = bbox
        cropped_object = source_img.crop((xmin, ymin, xmax, ymax))
        object_width = xmax - xmin
        object_height = ymax - ymin

        max_x = target_img.width - object_width
        max_y = target_img.height - object_height

        if max_x <= 0 or max_y <= 0:
            print(f"Object {bbox} is too large to paste on target image. Skipping...")
            continue

        # Ensure the object is not too tiny in the target image
        min_object_size = min(target_img.width, target_img.height) * 0.1
        if object_width < min_object_size or object_height < min_object_size:
            print(f"Object {bbox} is too small to be pasted on target image. Skipping...")
            continue

        valid_position_found = False
        max_retries = 50  # Limit retries

        for _ in range(max_retries):
            x_offset = random.randint(0, max_x)
            y_offset = random.randint(0, max_y)
            new_bbox = (x_offset, y_offset, x_offset + object_width, y_offset + object_height)

            overlap = False
            for existing_bbox in target_bboxes + new_bboxes:
                if calculate_iou(new_bbox, existing_bbox) > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                valid_position_found = True
                break

        if valid_position_found:
            # Apply photometric transformations
            transformed_object = apply_photometric_transformations(cropped_object)

            target_img.paste(transformed_object, (x_offset, y_offset))
            new_bboxes.append(new_bbox)
            pasted = True  # At least one object was successfully pasted
            print(f"Original bbox: {bbox}, New bbox: {new_bbox}")
        else:
            print(f"Could not find a valid position for object {bbox}. Skipping...")

    return target_img, new_bboxes, pasted

# Function to save updated annotation XML
def save_updated_xml(output_path, image_name, image_width, image_height, bboxes):
    try:
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "augmented_images"
        ET.SubElement(annotation, "filename").text = image_name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"

        for bbox in bboxes:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = "object"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(bbox[0])
            ET.SubElement(bndbox, "ymin").text = str(bbox[1])
            ET.SubElement(bndbox, "xmax").text = str(bbox[2])
            ET.SubElement(bndbox, "ymax").text = str(bbox[3])

        tree = ET.ElementTree(annotation)
        tree.write(output_path)
    except Exception as e:
        print(f"Error saving updated XML: {e}")

# Main function to perform Copy-Paste augmentation
def copy_paste_augmentation(images_folder, annotations_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    annotation_files = [f.replace('.jpg', '.xml') for f in image_files]

    augmented_images_count = 0
    max_augmented_images = 400

    while augmented_images_count < max_augmented_images:
        try:
            source_image_file = random.choice(image_files)
            source_annotation_file = source_image_file.replace('.jpg', '.xml')
            source_img = Image.open(os.path.join(images_folder, source_image_file))
            source_bboxes = parse_xml(os.path.join(annotations_folder, source_annotation_file))

            print(f"Source image: {source_image_file}")

            target_image_file = random.choice(image_files)
            target_annotation_file = target_image_file.replace('.jpg', '.xml')
            target_img = Image.open(os.path.join(images_folder, target_image_file))
            target_bboxes = parse_xml(os.path.join(annotations_folder, target_annotation_file))

            augmented_image, new_bboxes, pasted = paste_object_on_image_with_overlap_check(
                source_img, target_img, source_bboxes, target_bboxes
            )

            if pasted:  # Save only if at least one object was pasted
                combined_bboxes = target_bboxes + new_bboxes
                augmented_image_name = f'augmented_{os.path.splitext(source_image_file)[0]}_{os.path.splitext(target_image_file)[0]}.jpg'
                augmented_image_path = os.path.join(images_folder, augmented_image_name)
                augmented_image.save(augmented_image_path)

                augmented_xml_path = os.path.join(annotations_folder, augmented_image_name.replace('.jpg', '.xml'))
                save_updated_xml(
                    augmented_xml_path,
                    augmented_image_name,
                    target_img.width,
                    target_img.height,
                    combined_bboxes
                )

                augmented_images_count += 1
                print(f"Augmented image and annotation saved: {augmented_image_name}")
            else:
                print("No valid objects were pasted. Augmented image and annotation not saved.")
        except Exception as e:
            print(f"Error during augmentation: {e}")

# Paths to your images, annotations
images_folder = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\images\IP087"
annotations_folder = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\annotations\IP087"

# Perform the Copy-Paste augmentation
copy_paste_augmentation(images_folder, annotations_folder)
