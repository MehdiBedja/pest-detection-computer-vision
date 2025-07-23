# 🛠️ Dataset Preparation & Processing for Pest Detection

This folder contains scripts and utilities used to **prepare and preprocess the pest detection datasets** used in this project.  
The preprocessing workflow went through several iterations to ensure a **balanced, high-quality dataset** suitable for different object detection models.

---

## ✅ 1. Initial Exploration

- Started with the **IP102 dataset**, a well-known benchmark for insect pest classification & detection.
- Initially replicated a research study from Alexandria that focused on **five specific pests**.
- Created a **mini dataset** with only those pests by extracting images from IP102.
- Converted the dataset into:
  - **TFRecord format** for Faster R-CNN  
  - **YOLO-compatible format** for YOLOv7/8/9/11 experiments
- Trained multiple models to evaluate feasibility.

---

## ✅ 2. Selecting Pests Found in Algeria

- The goal was to choose pests **relevant to Algerian agriculture**.
- Requested the **crop-pest mapping list** directly from the IP102 authors.
- Selected pests affecting **wheat and other Algerian crops**.
- Verified their **presence in Algeria** using research papers & agricultural reports.
- Created a refined list of pests for a new dataset.

---

## ✅ 3. Addressing the Lack of Annotated Images

- Encountered **limited annotated images** (~100 images per pest) for the selected pests.
- Manually annotated **300 additional images** using a free online tool.
- Realized many **classification images were low-quality** and unsuitable for detection.
- Understood why the IP102 authors only annotated a subset for detection tasks.

---

## ✅ 4. Selecting Better-Represented Pests

- To ensure **enough training data**, wrote a script to **count detection annotations per pest**.
- Selected the **top 13 pests** with **400+ annotated detection images**.
- Focused on these well-represented pests for the final dataset.

---

## ✅ 5. Balancing the Dataset

- The 13 selected pest classes were **highly imbalanced**:
  - Some had only **400–500 images**
  - Others had **800–1000 images**
  - One class had **2900+ images**
  
To balance the dataset:
- For **underrepresented classes (<800 images)**:
  - Created a **copy-paste augmentation script**.
  - Extracted annotated objects & placed them into new images with constraints.
- For **overrepresented classes (>800 images)**:
  - Used **random sampling** to reduce them to **800 images**.

---

## ✅ 6. Final Dataset Preparation

- After balancing:
  - **13 pest classes**
  - **800 images per class** (total ~10,400 images)
- Converted the final dataset into:
  - **YOLO format** for YOLO models
  - **COCO/TFRecord format** for Faster R-CNN
- Finalized **train/val/test splits** with balanced class distribution.

---

## 🔄 Summary of Preprocessing Steps

1. **Filtered pests** → relevant to Algeria.  
2. **Counted image availability** → selected well-represented pests.  
3. **Manually annotated** missing detection images.  
4. **Balanced dataset** → augmentation for underrepresented, downsampling for overrepresented.  
5. **Converted formats** → YOLO & TFRecord for different models.  
6. **Created final dataset** → 13 pests × 800 images.

---