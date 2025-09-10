import os
import cv2
import numpy as np
import albumentations as A
import random
from pathlib import Path
import shutil
from collections import defaultdict
import yaml
from sklearn.model_selection import train_test_split

class HybridYOLOAugmentationPipeline:
    def __init__(self, images_path, labels_path, target_samples=800, max_multiplier=5):
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        self.target_samples = target_samples
        self.max_multiplier = max_multiplier
        
        # Create output directories
        self.base_output = self.images_path.parent / "processed_dataset"
        self.augmented_dir = self.base_output / "augmented"
        self.final_dir = self.base_output / "final_dataset"
        
        # Create directory structure
        (self.augmented_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.augmented_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Train/Val/Test directories
        for split in ["train", "val", "test"]:
            (self.final_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.final_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Class analysis
        self.class_counts = defaultdict(int)
        self.class_files = defaultdict(list)  # Store filenames per class
        self.deleted_classes = []
        
    def extract_class_from_filename(self, filename):
        """Extract class ID from IP102 filename format"""
        # Extract positions 3-5 (e.g., IP015000330.jpg -> 015)
        raw_class_id = int(filename[2:5])
        
        # Apply offset for classes 032 and above
        if raw_class_id >= 32:
            class_number = raw_class_id - 1
        else:
            class_number = raw_class_id
            
        return class_number
    
    def analyze_dataset(self):
        """Analyze current class distribution and group files by class"""
        print("Analyzing dataset...")
        
        for img_file in self.images_path.glob("*.jpg"):
            try:
                class_id = self.extract_class_from_filename(img_file.name)
                
                # Check if corresponding label exists
                label_file = self.labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    self.class_counts[class_id] += 1
                    self.class_files[class_id].append(img_file.stem)
                    
            except (ValueError, IndexError) as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
        
        print("Current class distribution:")
        for class_id, count in sorted(self.class_counts.items()):
            print(f"Class {class_id:03d}: {count} samples")
    
    def remove_small_classes(self, min_threshold=20):
        """Remove classes with insufficient samples"""
        classes_to_remove = [class_id for class_id, count in self.class_counts.items() 
                           if count < min_threshold]
        
        print(f"\n🗑️ Removing classes with <{min_threshold} samples:")
        for class_id in classes_to_remove:
            print(f"  - Class {class_id:03d}: {self.class_counts[class_id]} samples")
            del self.class_counts[class_id]
            del self.class_files[class_id]
            
        self.deleted_classes = classes_to_remove
        return classes_to_remove
    
    def create_augmentation_pipeline(self, severity='heavy'):
        """Create augmentation pipeline based on severity level"""
        
        if severity == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.3
                ),
                A.Blur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(5, 15), p=0.2),
                A.Rotate(limit=5, p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        elif severity == 'heavy':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.4),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.4
                ),
                A.OneOf([
                    A.Blur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.GaussNoise(var_limit=(10, 25), p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transform
    
    def calculate_augmentation_strategy(self):
        """Calculate smart augmentation strategy"""
        strategy = {}
        
        print(f"\n📊 Augmentation Strategy (Target: {self.target_samples} samples):")
        print("="*70)
        
        for class_id, count in sorted(self.class_counts.items()):
            if count >= self.target_samples:
                # No augmentation needed
                strategy[class_id] = {
                    'severity': 'none',
                    'augmentations_per_image': 0,
                    'total_needed': 0,
                    'final_count': count
                }
                print(f"Class {class_id:03d}: {count:4d} → {count:4d} (no augmentation)")
                
            else:
                # Calculate how many we need
                needed = self.target_samples - count
                
                # Apply max multiplier constraint
                max_possible = count * self.max_multiplier
                if needed > max_possible:
                    actual_needed = max_possible
                    final_count = count + actual_needed
                else:
                    actual_needed = needed
                    final_count = self.target_samples
                
                augs_per_image = actual_needed // count if count > 0 else 0
                
                # Choose severity based on current count
                if count < 200:
                    severity = 'heavy'  
                else:
                    severity = 'light'
                
                strategy[class_id] = {
                    'severity': severity,
                    'augmentations_per_image': augs_per_image,
                    'total_needed': actual_needed,
                    'final_count': final_count
                }
                
                print(f"Class {class_id:03d}: {count:4d} → {final_count:4d} "
                      f"(+{actual_needed:3d}, {augs_per_image}x aug, {severity})")
        
        return strategy
    



    #augmentation happen here 205 hereeeeeeeeeeeeee
    def augment_image_and_labels(self, img_name, transform, num_augmentations):
        """Augment a single image and its labels"""
        img_path = self.images_path / f"{img_name}.jpg"
        label_path = self.labels_path / f"{img_name}.txt"
        
        if not img_path.exists() or not label_path.exists():
            return []
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load bounding boxes
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        # Generate augmentations
        augmented_files = []
        
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                augmented = transform(
                    image=image, 
                    bboxes=bboxes, 
                    class_labels=class_labels
                )
                
                # Save augmented image
                aug_img_name = f"{img_name}_aug_{i}.jpg"
                aug_img_path = self.augmented_dir / "images" / aug_img_name
                
                aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)
                
                # Save augmented labels
                aug_label_name = f"{img_name}_aug_{i}.txt"
                aug_label_path = self.augmented_dir / "labels" / aug_label_name
                
                with open(aug_label_path, 'w') as f:
                    for bbox, class_label in zip(augmented['bboxes'], augmented['class_labels']):
                        f.write(f"{class_label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                augmented_files.append((aug_img_name, aug_label_name))
                
            except Exception as e:
                print(f"Error augmenting {img_name}: {e}")
                continue
        
        return augmented_files
    







    def run_augmentation(self):
        """Run the complete augmentation pipeline"""
        print("🚀 Starting Hybrid Augmentation Pipeline...")
        
        # Step 1: Analyze dataset
        self.analyze_dataset()
        
        # Step 2: Remove small classes
        deleted_classes = self.remove_small_classes()
        
        # Step 3: Calculate strategy
        strategy = self.calculate_augmentation_strategy()
        
        # Step 4: Apply augmentations
        print(f"\n🔄 Applying augmentations...")
        
        for class_id, strat in strategy.items():
            if strat['augmentations_per_image'] == 0:
                continue
                
            print(f"Processing Class {class_id:03d} with {strat['severity']} augmentation...")
            transform = self.create_augmentation_pipeline(strat['severity'])
            
            # Get files for this class
            class_files = self.class_files[class_id]
            
            for img_name in class_files:
                try:
                    self.augment_image_and_labels(
                        img_name, transform, strat['augmentations_per_image']
                    )
                except Exception as e:
                    print(f"Failed to augment {img_name}: {e}")
        
        print("✅ Augmentation complete!")
        
        # Step 5: Create final dataset with train/val/test split
        self.create_final_dataset_with_splits()
        
        # Step 6: Calculate and save class weights
        self.calculate_class_weights()
    
    def create_final_dataset_with_splits(self):
        """Combine original and augmented data, then split into train/val/test"""
        print("\n📁 Creating final dataset with train/val/test splits...")
        
        all_files = []
        
        # Collect all original files (excluding deleted classes)
        for img_file in self.images_path.glob("*.jpg"):
            try:
                class_id = self.extract_class_from_filename(img_file.name)
                if class_id not in self.deleted_classes:
                    label_file = self.labels_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        all_files.append((img_file, label_file, "original"))
            except:
                continue
        
        # Collect all augmented files
        for img_file in (self.augmented_dir / "images").glob("*.jpg"):
            label_file = self.augmented_dir / "labels" / f"{img_file.stem}.txt"
            if label_file.exists():
                all_files.append((img_file, label_file, "augmented"))
        
        print(f"Total files to split: {len(all_files)}")
        
        # Split: 70% train, 10% val, 20% test
        train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.667, random_state=42)  # 0.667 of 0.3 = 0.2
        
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        # Copy files to respective splits
        for split_name, files in splits.items():
            print(f"Creating {split_name} set: {len(files)} files")
            
            for img_file, label_file, source in files:
                # Copy image
                dst_img = self.final_dir / split_name / "images" / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy label  
                dst_label = self.final_dir / split_name / "labels" / label_file.name
                shutil.copy2(label_file, dst_label)
        
        print(f"✅ Final dataset created at: {self.final_dir}")
        print(f"   📂 Train: {len(train_files)} files ({len(train_files)/len(all_files)*100:.1f}%)")
        print(f"   📂 Val:   {len(val_files)} files ({len(val_files)/len(all_files)*100:.1f}%)")
        print(f"   📂 Test:  {len(test_files)} files ({len(test_files)/len(all_files)*100:.1f}%)")
    
    def calculate_class_weights(self):
        """Calculate and save class weights based on final distribution"""
        print("\n⚖️ Calculating class weights...")
        
        # Count final distribution from train set
        final_counts = defaultdict(int)
        train_labels_dir = self.final_dir / "train" / "labels"
        
        for label_file in train_labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        final_counts[class_id] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = sum(final_counts.values())
        num_classes = len(final_counts)
        
        class_weights = {}
        for class_id, count in final_counts.items():
            # Standard inverse frequency weighting
            weight = total_samples / (num_classes * count)
            class_weights[class_id] = round(weight, 4)
        
        # Save weights to YAML file
        weights_file = self.final_dir / "class_weights.yaml"
        
        weights_info = {
            'class_weights': class_weights,
            'final_distribution': dict(final_counts),
            'total_samples': total_samples,
            'num_classes': num_classes,
            'deleted_classes': self.deleted_classes,
            'usage': 'Use these weights in your YOLO training config: cls: <weight_value> for each class'
        }
        
        with open(weights_file, 'w') as f:
            yaml.dump(weights_info, f, default_flow_style=False)
        
        print("📊 Final Class Distribution (Train Set):")
        print("="*50)
        for class_id, count in sorted(final_counts.items()):
            weight = class_weights[class_id]
            print(f"Class {class_id:03d}: {count:4d} samples (weight: {weight:.4f})")
        
        print(f"\n💾 Class weights saved to: {weights_file}")
        print(f"📋 Deleted classes: {self.deleted_classes}")
        
        return class_weights

# Usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = HybridYOLOAugmentationPipeline(
        images_path="F:/code_pfe_all/IP102_DATASET/annotated_images/etud/dataset_before/images",
        labels_path="F:/code_pfe_all/IP102_DATASET/annotated_images/etud/dataset_before/labels", 
        target_samples=800,  # Target samples per class
        max_multiplier=5     # Max 5x augmentation for small classes
    )
    
    # Run the complete pipeline
    pipeline.run_augmentation()
    
    print("\n🎉 Pipeline completed successfully!")
    print("📁 Check the 'processed_dataset/final_dataset' folder for train/val/test splits")
    print("📋 Check 'class_weights.yaml' for recommended training weights")