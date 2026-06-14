# Campus Concentration Monitoring System

A real-time AI-driven computer vision system that monitors student concentration levels (attentive, sleepy, or distracted) and performs real-time face counting.

---

## Mandatory Requirements

#### Data
* Minimum 3 classes: Attentive, Distracted, Drowsy (Sleepy)
* Minimum **200 annotations/images per class** (600+ total)
* Self-collected dataset using webcam or recorded lectures
* Annotation format: YOLO .txt or COCO JSON

#### Input
* RGB Images or Video Frames
* Classroom scenes with single or multiple students.

#### Output
* Annotated frames with bounding boxes + engagement label per person
* Engagement summary statistics (e.g., % attentive per session)

#### AI Pipelines
* Deep Learning or Machine Learning
* Object classification
* Transfer Learning

![Architecture Draft](ArchitectureDraft.png)

---

## Features

- **Real-Time Attention Tracking**: Classifies student states as `ATTENTIVE`, `SLEEPY`, or `DISTRACTED` using eye closure (EAR/PERCLOS), yawning rate (MAR), and head pose (Pitch, Yaw, Roll).
- **Unified DearPyGui Dashboard**: Full-featured dashboard with live webcam/video/image input, real-time face detection, and a runtime-swappable classifier dropdown.
- **Six Classifiers**: Rule-based, MLP, MobileNetV3, EfficientNetV2, DeiT-Tiny, and MobileViT-XXS — all selectable without restarting.
- **Face Counter Dashboard**: Tkinter-based dashboard showing live face counts, session duration, peak faces, and FPS using YOLOv8.
- **Attention Pipeline**: Lightweight CLI pipeline for landmark-based classification (rule-based or MLP).

---

## Classifiers

| Classifier | Type | Input | Notes |
|---|---|---|---|
| Rule-based | Threshold rules | Landmarks | No training required |
| MLP | sklearn MLP | 9 landmark features | Requires trained `mlp_model.pkl` |
| MobileNetV3 | CNN (timm) | 224×224 face crop | `best_mobilenetv3_with.pth` |
| EfficientNetV2 | CNN (timm) | 224×224 face crop | `best_efficientnetv2_with.pth` |
| DeiT-Tiny | Vision Transformer | 224×224 face crop | `best_deit_tiny_with.pth` |
| MobileViT-XXS | Hybrid ViT | 256×256 face crop | `best_mobilevit_xxs_with.pth` |

---

## Installation & Setup

> **Python 3.11 required** — MediaPipe has strict version requirements.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Benji1928/Campus_Concentration_Monitoring_System.git
   cd Campus_Concentration_Monitoring_System
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install optional extras** (required for specific entry points):
   ```bash
   pip install ultralytics pillow   # face_counter.py
   pip install dearpygui            # src/dashboard.py
   ```

---

## How to Run

### 1. Unified Dashboard (DearPyGui)

Full dashboard with YOLOv8 face detection and runtime-swappable classifier. Supports webcam, video files, and images.

```bash
python src/dashboard.py
```

**Controls:**
- **Source** — toggle between Webcam and File
- **cam** — camera index (default `0`)
- **Browse** — select an image or video file
- **Start / Stop** — begin or halt inference
- **Classifier dropdown** — switch between all six classifiers without restarting

The results panel shows the predicted label, confidence, per-class probability bars, and live feature values (EAR, MAR, Yaw, Pitch, PERCLOS, Blink/min).

---

### 2. Attention Monitoring Pipeline

Lightweight CLI pipeline using MediaPipe face landmarks. Downloads the Google Face Landmarker model (~30 MB) on first run.

- **Rule-based only**:
  ```bash
  python src/pipeline.py
  ```

- **Rule-based + MLP side-by-side** *(requires trained MLP model)*:
  ```bash
  python src/pipeline.py --mlp
  ```

- **Specific camera index**:
  ```bash
  python src/pipeline.py --camera 1
  ```

Press **`Q`** to exit.

---

### 3. Face Counter Dashboard

Tkinter-based dashboard showing live face counts, session duration, peak faces, and FPS.

- **Default settings**:
  ```bash
  python src/face_counter.py
  ```

- **Custom model and confidence**:
  ```bash
  python src/face_counter.py --model models/face_detection.pt --conf 0.75 --cam 0
  ```

Press **`Q`** or **`ESC`** to quit.

---

## Training

### MLP Classifier

Collect landmark features to `data/labeled_features.csv` (columns: `FEATURE_COLS` + `label`), then:

```bash
python src/training/train.py
```

Outputs: `models/mlp_model.pkl`, `models/scaler.pkl`

### CNN / Transformer Classifiers

Training scripts live in `src/classifiers/classification_pipeline/`:

| Script | Model |
|---|---|
| `CNN_training_v1.py` | EfficientNet-B0 |
| `CNN_training_v2.py` | MobileNetV3-Large |
| `Transformer_training_deit.py` | DeiT-Tiny |
| `Transformer_training_mobilevit.py` | MobileViT-XXS |

Configs are in `cnn_configs.py` and `transformer_configs.py`. Set `dataset_dir` to your local dataset path before running.

---

## Architecture

### Data Flow — Unified Dashboard

```
YOLOv8 face detection
  → largest face bbox → crop (+ 10% padding)
  → [if landmark classifier] MediaPipe FaceMesh → FeatureExtractor (9 features)
  → active classifier.predict(face_crop, features)
  → ClassifierResult(label, confidence, probabilities)
```

### Data Flow — Attention Pipeline

```
MediaPipe FaceMesh → 468 landmarks
  → FeatureExtractor.extract() → 9-feature dict
  → RuleBasedClassifier / MLPAttentionClassifier
```

### Models Directory

| File | Used by |
|---|---|
| `face_detection.pt` | YOLOv8 face detector |
| `face_landmarker.task` | MediaPipe FaceMesh |
| `mlp_model.pkl` + `scaler.pkl` | MLP classifier |
| `best_mobilenetv3_with.pth` | MobileNetV3 classifier |
| `best_efficientnetv2_with.pth` | EfficientNetV2 classifier |
| `best_deit_tiny_with.pth` | DeiT-Tiny classifier |
| `best_mobilevit_xxs_with.pth` | MobileViT-XXS classifier |
