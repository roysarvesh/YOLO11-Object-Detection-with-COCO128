#VisionX: Real-Time Multi-Object 

A beginner-friendly, end-to-end project to learn **Object Detection using Ultralytics YOLO11** and the **COCO128** dataset. This repository is designed as a hands-on learning project that covers the complete computer vision pipeline—from environment setup and dataset exploration to model training, evaluation, and real-time webcam inference.

Unlike a simple tutorial, this project focuses on understanding **how YOLO works internally**, including the dataset format, model architecture, training workflow, evaluation metrics, and deployment.

---

# 📖 Project Objectives

By completing this project, you will learn:

* Python project structure
* Virtual environments and dependency management
* YOLO dataset format
* COCO128 dataset organization
* Data annotation format
* Transfer Learning
* YOLO11 architecture
* Model training
* Model evaluation
* Image inference
* Webcam object detection
* Performance metrics (Precision, Recall, mAP)

---

# 📂 Project Structure

```text
YOLO_COCO128/
│
├── data/
│   └── coco128/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── train.py
├── predict.py
├── webcam.py
├── evaluate.py
├── explore_dataset.py
├── load_model.py
├── requirements.txt
└── README.md
```

---

# 🛠 Requirements

* Python 3.10+
* Git
* VS Code (Recommended)

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/YOLO_COCO128.git

cd YOLO_COCO128
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install ultralytics
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install pandas
pip install numpy
pip install jupyter
```

Or simply:

```bash
pip install -r requirements.txt
```

---

## 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

Run:

```bash
yolo checks
```

This verifies that Ultralytics and PyTorch are installed correctly.

---

# 📁 Dataset

The project uses the **COCO128** dataset provided by Ultralytics.

Download it automatically by running:

```bash
yolo detect train model=yolo11n.pt data=coco128.yaml epochs=1
```

The dataset will be downloaded into:

```text
datasets/
└── coco128/
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    │
    ├── labels/
    │   ├── train2017/
    │   └── val2017/
    │
    └── coco128.yaml
```

---

# 🏷 YOLO Annotation Format

Each image has a corresponding `.txt` annotation file.

Example:

```text
0 0.47 0.52 0.25 0.34
```

Meaning:

```text
Class_ID
Center_X
Center_Y
Width
Height
```

All coordinates are normalized between **0 and 1**.

---

# 🔍 Exploring the Dataset

Run:

```bash
python explore_dataset.py
```

This script counts the images inside the training directory.

You can also visualize images using OpenCV.

---

# 🧠 Understanding the YOLO11 Model

Load the pretrained model:

```bash
python load_model.py
```

This prints the complete YOLO11 neural network architecture.

High-level architecture:

```text
Input Image
      │
      ▼
Backbone
      │
      ▼
Neck
      │
      ▼
Detection Head
      │
      ▼
Bounding Boxes
```

---

# 🎯 Training the Model

Run:

```bash
python train.py
```

The model is trained using:

* YOLO11 Nano (`yolo11n.pt`)
* COCO128 Dataset
* Image Size: 640 × 640
* Batch Size: 16
* 50 Epochs (recommended if using a GPU)

Training workflow:

```text
Read Image
      │
Resize
      │
Data Augmentation
      │
Tensor Conversion
      │
Forward Pass
      │
Prediction
      │
Loss Calculation
      │
Backpropagation
      │
Optimizer Step
      │
Next Batch
```

---

# 📊 Model Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

The evaluation reports:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

---

# 🖼 Image Prediction

Run:

```bash
python predict.py
```

Example prediction:

```python
model.predict(
    source="test.jpg",
    save=True,
    conf=0.25
)
```

The output is saved in:

```text
runs/detect/predict/
```

---

# 📷 Webcam Detection

Start real-time object detection:

```bash
python webcam.py
```

YOLO detects objects directly from your webcam.

---

# 📁 Training Outputs

After training, Ultralytics automatically creates:

```text
runs/
└── detect/
    └── train/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        │
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        ├── PR_curve.png
        ├── F1_curve.png
        └── args.yaml
```

---

# 🔄 Complete Training Pipeline

```text
Dataset
      │
      ▼
Image Loading
      │
      ▼
Resize & Augmentation
      │
      ▼
Tensor Conversion
      │
      ▼
YOLO11 Backbone
      │
      ▼
Neck (Feature Fusion)
      │
      ▼
Detection Head
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Optimizer
      │
      ▼
Validation
      │
      ▼
Save best.pt
      │
      ▼
Inference
```


---

# 📄 License

This project is intended for educational purposes and follows the licensing terms of the Ultralytics YOLO framework and the COCO dataset.

---

## ⭐ If this project helped you learn YOLO, consider giving the repository a star!
