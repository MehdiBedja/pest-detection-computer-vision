# 📂 Data Overview

This folder provides **links and documentation** for all datasets used in this project.  
Since the datasets are large, they are hosted on **Kaggle** for easy access.

The project experiments were based on **IP102**, a large-scale agricultural pest dataset.  
From this, a **filtered subset of 13 pest classes** was created and preprocessed for different object detection models (YOLO & Faster R-CNN).

---

## ✅ 1. IP102 Full Dataset (YOLO-ready format)

- **Description:**  
  The complete **IP102 dataset** with **102 pest species**.  
  Already converted into **YOLOv5/YOLOv8-compatible format** for direct training.


- **Kaggle Link:**  
  🔗 [Download IP102 YOLO Format Dataset](<https://www.kaggle.com/datasets/mehdi3333/yolo-full-dataset>)  

---

## ✅ 2. Filtered Dataset: 13 Selected Pest Classes
- **Description:**  
  A **cleaned & filtered subset** of the IP102 dataset containing **only 13 agriculturally important pest classes**.  
  Images were selected based on quality, relevance, and class balance.



- **Kaggle Link:**  
  🔗 [Download Preprocessed 13-Class Dataset](<https://www.kaggle.com/datasets/bedjamahdi/pest-dataset-with-chosen-classes-before-treatment>)  

---

## ✅ 3. YOLO-Formatted Dataset AFTER PREPROCESSING (13 Classes)

- **Description:**  
  The **13-class dataset**, converted into **YOLO-friendly format**.  
  Fully compatible with **YOLOv7, YOLOv8, YOLOv9, YOLOv11** for training & evaluation.

- **Kaggle Link:**  
  🔗 [Download YOLO 13-Class Dataset](<https://www.kaggle.com/datasets/bedjamahdi/dataset-yolo-v1-0-13pesttypes>)  

---

## ✅ 4. Faster R-CNN SSD Dataset AFTER PREPROCESSING (13 Classes)

- **Description:**  
  The same **13-class dataset**, converted into a **TFRecord/COCO-compatible format** for use with **Faster R-CNN** and other TensorFlow-based detectors.

- **Kaggle Link:**  
  🔗 [Download Faster R-CNN 13-Class Dataset](<https://www.kaggle.com/datasets/bedjamahdi/dataset13pesttfrecords>)  

---

