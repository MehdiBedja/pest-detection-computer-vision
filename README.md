# A-Mobile-Application-for-Agricultural-Pest-Recognition
 Pest Detection Using Deep Learning & Mobile Deployment
🚀 From dataset curation → model training → evaluation → real-world deployment on a mobile app

This project is an end-to-end computer vision system for agricultural pest detection, built to assist farmers in identifying harmful pests in crops. It integrates deep learning models, dataset engineering, custom preprocessing pipelines, and full mobile deployment.

✅ Core Highlights

Dataset engineering with multiple preprocessing steps, balancing & augmentation

Trained multiple state-of-the-art models (YOLOv7 → YOLOv8 → YOLOv9 → YOLOv10 → YOLOv11, SSD, Faster R-CNN)

Hyperparameter tuning & architecture tweaking to optimize results

Final model selection: YOLOv11 (best trade-off of accuracy vs. performance)
<img src="deployment\testImageResult.jpg" width="400">

TensorFlow Lite conversion for real-time on-device inference

Deployed in a fully functional mobile app, with backend & frontend integration
<img src="deployment\resultMobileAPP.jpg" width="400">


🔗 Mobile App Frontend Repository : https://github.com/MehdiBedja/pest_detection_app_frontend
🔗 Backend Repository :    https://github.com/MehdiBedja/pest_detection_app_backend

 Project Overview
This project addresses the lack of robust pest recognition tools in rural areas by providing a fully offline-capable pest detection system.

✅ What it does:
Detects 13 critical agricultural pests (More in the future)

Works offline on smartphones using a TFLite-optimized model

Displays bounding boxes + pest class names

Maps detected pests to recommended pesticides & details 

 Dataset Engineering & Preprocessing
This project heavily relies on IP102, the largest benchmark dataset for agricultural pests. However, extensive preprocessing was required to adapt it for real-world deployment.

(Details of preprocessing steps in data/README.md)

 Models Trained & Evaluated
I iteratively trained and compared multiple object detection architectures to find the optimal trade-off:

✅ YOLOv7 (baseline)

✅ YOLOv8 (s, m) (improved accuracy)

✅ YOLOv9 & YOLOv10 (tested latest versions)

✅ YOLOv11s (final) (best performance on dataset)

✅ SSD MobileNet v2 (lightweight baseline)

✅ Faster R-CNN ResNet50 (high accuracy, but heavier)

Each model was evaluated on the same test set, with mAP@50 & mAP@50-95, precision, and recall reported.

 Detailed results & charts are available in the results/ folder.

 Final Model Selection: YOLOv11s
Achieved mAP@50 = 93.8%
<img src="results\yolov11s\Screenshot 2025-02-12 153319.png" width="400">
<img src="results\yolov11s\confusion_matrix_normalized.png" width="400">
<img src="results\yolov11s\confusion_matrix.png" width="400">
<img src="results\yolov11s\P_curve.png" width="400">
<img src="results\yolov11s\PR_curve.png" width="400">


Converted to TensorFlow Lite for mobile deployment


Repository Structure

pest-detection-computer-vision/
│
├── data/                   # Dataset links
└── deployment/             # detect.py + mobile-ready models
├── preprocessing/          # preprocessing steps
├── results/                # Metrics, charts & sample results
├── training_notebooks/     # Jupyter notebooks for training




This project shows I can handle data engineering, deep learning research, model optimization, and production deployment all in one pipeline.

I’m continuing to tweak model architectures for even better results, which will be shared in future updates.

results/charts/  results/sample_results/ 
