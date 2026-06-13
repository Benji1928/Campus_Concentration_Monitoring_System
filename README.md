<<<<<<< Updated upstream
# Campus_Concentration_Monitoring_System
Create AI, check Students if "a mimir" or locked in
=======
# Campus Concentration Monitoring System

A real-time AI-driven computer vision system designed to monitor student concentration levels (attentive, sleepy, or distracted) and perform real-time face counting.

---

## Mandatory Requirements

- [ ]
#### Data
* Minimum 3 classes: Attentive, Distracted, Drowsy (Sleepy)
* Minimum **200 annotations/images per class** (600+ total)
* Self-collected dataset using webcam or recorded lectures
* Annotation format: YOLO .txt or COCO JSON

- [ ]
#### Input
* RGB Images or Video Frames
* Classroom scenes with single or multiple students.

- [ ]
#### Output
* Annotated frames with bounding boxes + engagement label per person
* Engagement summary statistics (e.g., % attentive per session)

- [ ]
#### AI Pipelines
* Deep Learning or Machine Learning
* Object classification
* Transfer Learning 

![alt text](ArchitectureDraft.png)

---

## Features
- **Real-Time Attention Tracking**: Classify student states as `ATTENTIVE`, `SLEEPY`, or `DISTRACTED` using eye closure levels (EAR/PERCLOS), yawning rates (MAR), and head pose tracking (Pitch, Yaw, Roll).
- **Dual Classifiers**: Supports both deterministic **Rule-Based** and machine-learning-based **MLP (Multi-Layer Perceptron)** classifiers.
- **Face Counter Dashboard**: A Tkinter-based dashboard that detects and counts faces in real-time using a YOLOv8 face detection model.

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Benji1928/Campus_Concentration_Monitoring_System.git
   cd Campus_Concentration_Monitoring_System
   ```

2. **Create and Activate a Virtual Environment**:
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

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Attention Monitoring Pipeline
Run the real-time pipeline to detect face landmarks and output concentration status. On the first run, the script will automatically download the required Google MediaPipe Face Landmarker model (~30 MB).

- **Run using only the Rule-Based Classifier**:
  ```bash
  python src/pipeline.py
  ```

- **Run with both Rule-Based and MLP Classifiers (Side-by-Side)**:
  *(Note: Requires the MLP model to be trained first, see training instructions below)*
  ```bash
  python src/pipeline.py --mlp
  ```

- **Run on a specific camera index (e.g., secondary external camera)**:
  ```bash
  python src/pipeline.py --camera 1
  ```

Press **`Q`** inside the webcam feed window to exit.

---

### 2. Face Counter Dashboard
Run the Tkinter-based dashboard displaying face counts, session durations, peak faces, and frame-rate (FPS).

- **Run with default settings**:
  ```bash
  python src/face_counter.py
  ```

- **Run with custom YOLO model and confidence threshold**:
  ```bash
  python src/face_counter.py --model models/face_detection.pt --conf 0.75 --cam 0
  ```

Press **`Q`** or **`ESC`** to quit the dashboard.

---

### 3. Training the MLP Classifier
If you collect feature datasets under `data/labeled_features.csv`, you can train the MLP network:

```bash
python src/training/train.py
```

This will split the data, scale the features, train a multi-layer neural network, print a classification performance report, and save the model assets:
- `models/mlp_model.pkl`
- `models/scaler.pkl`


>>>>>>> Stashed changes
