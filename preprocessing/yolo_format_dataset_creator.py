import os
import shutil
import random
import xml.etree.ElementTree as ET

# Define paths
images_root = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\images"
annotations_root = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\annotations"
output_root = r"F:\code_pfe_all\IP102_DATASET\dataset\yolo_dataset"

# YOLO Format Helper Function
def convert_to_yolo(size, bbox):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[1]) / 2.0 - 1
    y = (bbox[2] + bbox[3]) / 2.0 - 1
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    return x * dw, y * dh, w * dw, h * dh

# Parse XML to YOLO
def parse_xml_to_yolo(xml_path, class_index):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    yolo_annotations = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        yolo_bbox = convert_to_yolo((width, height), (xmin, xmax, ymin, ymax))
        yolo_annotations.append(f"{class_index} " + " ".join(map(str, yolo_bbox)))

    return yolo_annotations

# Dataset Split Function
def split_dataset(images_root, annotations_root, output_root, split_ratios=(0.8, 0.1, 0.1)):
    classes = os.listdir(images_root)
    splits = ['train', 'val', 'test']

    # Create output directories
    for split in splits:
        os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)

    # Process each class
    for class_index, class_name in enumerate(classes):
        image_class_dir = os.path.join(images_root, class_name)
        annotation_class_dir = os.path.join(annotations_root, class_name)

        images = [f for f in os.listdir(image_class_dir) if f.endswith('.jpg')]

        # Shuffle and split
        random.shuffle(images)
        train_count = int(len(images) * split_ratios[0])
        val_count = int(len(images) * split_ratios[1])

        datasets = {
            'train': images[:train_count],
            'val': images[train_count:train_count + val_count],
            'test': images[train_count + val_count:]
        }

        # Move and convert
        for split, split_images in datasets.items():
            for image_name in split_images:
                # Copy image
                src_image_path = os.path.join(image_class_dir, image_name)
                dst_image_path = os.path.join(output_root, split, 'images', image_name)
                shutil.copy(src_image_path, dst_image_path)

                # Convert and save annotation
                annotation_name = image_name.replace('.jpg', '.xml')
                src_annotation_path = os.path.join(annotation_class_dir, annotation_name)
                if os.path.exists(src_annotation_path):
                    yolo_annotations = parse_xml_to_yolo(src_annotation_path, class_index)
                    dst_annotation_path = os.path.join(output_root, split, 'labels', image_name.replace('.jpg', '.txt'))
                    with open(dst_annotation_path, 'w') as f:
                        f.write("\n".join(yolo_annotations))

        # Print the counts for this class
        print(f"Class '{class_name}' -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

# Run the split
split_dataset(images_root, annotations_root, output_root)
