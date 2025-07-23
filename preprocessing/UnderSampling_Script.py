import os
import random

# Paths to dataset (update with your paths)
images_folder = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\images"
annotations_folder = r"F:\code_pfe_all\IP102_DATASET\dataset\v1.1\annotations"

# Classes to undersample
classes_to_undersample = ["IP052", "IP071", "IP102"]  # Replace with actual folder names
target_samples = 870

# Function to undersample
def undersample_class(images_path, annotations_path, target_count):
    all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    all_annotations = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]

    # Ensure matching images and annotations
    images_set = set(os.path.splitext(f)[0] for f in all_images)
    annotations_set = set(os.path.splitext(f)[0] for f in all_annotations)
    matched_files = list(images_set.intersection(annotations_set))

    if len(matched_files) > target_count:
        # Shuffle and select target samples
        random.shuffle(matched_files)
        retained_files = matched_files[:target_count]
        removed_files = matched_files[target_count:]

        # Delete excess images and annotations
        for file_base in removed_files:
            image_path = os.path.join(images_path, file_base + '.jpg')  # Adjust extension if needed
            annotation_path = os.path.join(annotations_path, file_base + '.xml')

            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(annotation_path):
                os.remove(annotation_path)

        print(f"Reduced class to {target_count} samples at {images_path} and {annotations_path}.")
    else:
        print(f"Class already has {len(matched_files)} samples or fewer. No action needed.")

# Apply undersampling to each class
for class_name in classes_to_undersample:
    class_images_path = os.path.join(images_folder, class_name)
    class_annotations_path = os.path.join(annotations_folder, class_name)
    undersample_class(class_images_path, class_annotations_path, target_samples)
