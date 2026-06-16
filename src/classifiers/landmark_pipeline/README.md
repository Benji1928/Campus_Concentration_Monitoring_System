# Landmark Pipeline

One of two pipelines in this project. Classifies student engagement states — **Attentive**, **Sleepy**, or **Distracted** — using facial landmark geometry and a trained **Multi-Layer Perceptron (MLP)** neural network.

---

## How It Works

```
Webcam → MediaPipe (468 landmarks) → Feature Extraction (9 features) → MLP / Rule-Based Classifier → Label
```

### Extracted Features

| Feature | Description |
|---|---|
| `ear_left`, `ear_right`, `ear_avg` | Eye Aspect Ratio — measures eye openness |
| `mar` | Mouth Aspect Ratio — detects yawning |
| `pitch`, `yaw`, `roll` | Head orientation in degrees via solvePnP |
| `perclos` | Fraction of frames with eyes closed over a 3-second rolling window |
| `blink_rate` | Blinks per minute over a 60-second rolling window |

---

## Dataset Used:

The two datasets links:
* https://www.kaggle.com/datasets/shivampandey1233/drowsy-dataset
* https://universe.roboflow.com/neurosense/user-attention
* https://universe.roboflow.com/distractless/distractless
* 


Potential Test Dataset:
* https://universe.roboflow.com/bklab/students-in-lecture
* https://universe.roboflow.com/123-cpztz/ml-pjutg

---

## Installation

```powershell
pip install "numpy<2.0"
pip install mediapipe opencv-python scikit-learn pandas joblib
```

---

## Step 1 — Collect Training Data

```powershell
python src/classifiers/landmark_pipeline/collect_dataset.py
```  
Reads images from data/kaggle_drowsy/ + Roboflow folders, runs MediaPipe, writes data/labeled_features.csv. **You might need to change the API Key to ROBOFLOW, set to nothing, "" to not auto-download the dataset**


#### Optional: Run the data collector and hold keys to label frames in real time.

```powershell
python src/classifiers/landmark_pipeline/collector.py
```

**Controls:**

| Key | Label |
|---|---|
| Hold `1` | ATTENTIVE |
| Hold `2` | SLEEPY |
| Hold `3` | DISTRACTED |
| Press `Q` | Quit and save |

Saves to `data/labeled_features.csv`. Run multiple sessions — data appends each time. Target **200+ samples per class**.

---

## Step 2 — Train the MLP

Once `data/labeled_features.csv` exists, either:

**Option A — Run locally:**
```powershell
python src/classifiers/landmark_pipeline/train.py
```

**Option B — Run on Google Colab:**

Upload `labeled_features.csv` to Google Drive, then open and run:
```
src/classifiers/landmark_pipeline/MLP_training.ipynb
```

Both options produce `models/mlp_model.pkl` and `models/scaler.pkl`.

---

## Step 3 - Check Model against Ground Truth (Evaluation)

Open:

* model_evaluation.ipynb

Run and check the cell output.

---

## Step 4 — Run the Pipeline

**Rule-based only (no training required):**
```powershell
python src/classifiers/landmark_pipeline/pipeline.py
```

**Rule-based + MLP side-by-side:**
```powershell
python src/classifiers/landmark_pipeline/pipeline.py --mlp
```

**Alternate camera:**
```powershell
python src/classifiers/2223_pipeline/pipeline.py --camera 1
```

Press `Q` to quit.

---

## File Reference

| File | Purpose | Key Functions |
|---|---|---|
| `pipeline.py` | Main entry point — runs live webcam feed | `main()` — webcam loop, draws HUD with label and metrics |
| `collector.py` | Data collection tool — records labeled training data | `main()` — hold 1/2/3 to label frames, saves to `data/labeled_features.csv` |
| `face_mesh.py` | Detects 468 facial landmarks via MediaPipe | `FaceMesh.process(frame)` — returns landmark coordinates or `None` |
| `feature_extractor.py` | Computes 9 behavioral features from landmarks | `FeatureExtractor.extract(landmarks)` — returns dict of EAR, MAR, head pose, PERCLOS, blink rate |
| `rule_based.py` | Classifies engagement using hardcoded thresholds | `RuleBasedClassifier.predict(features)` — returns label string and int |
| `mlp_classifier.py` | MLP neural network classifier | `fit()`, `predict()`, `predict_proba()`, `save()`, `load()` |
| `train.py` | Trains MLP from CSV, prints report, saves model | `main()` — loads CSV → trains → evaluates → saves `.pkl` files |
| `MLP_training.ipynb` | Colab notebook — full training and evaluation pipeline | EDA, preprocessing, training, loss curve, confusion matrix, save model |



---

## Checklist

- [ ] Install dependencies
- [ ] Collect 200+ samples per class using `collector.py`
- [ ] Train MLP via `train.py` or Colab notebook
- [ ] Run `pipeline.py --mlp` and verify classifications

---

## File Functionalities:
Here is your file structure converted into a clean, organized Markdown format. I've structured it as a directory tree and a detailed breakdown table for maximum readability.

### Project Structure

```text
├── collect_dataset.py       # Step 1: Extract features from datasets
├── train_evaluate.py        # Step 2: Train models + generate all graphs
├── model_evaluation.ipynb   # Step 3: Ground truth vs. predicted analysis
├── pipeline.py              # Step 4: Real-time webcam inference
├── feature_extractor.py     # Core: Computes the 9 features (used by collect & pipeline)
├── face_mesh.py             # Core: MediaPipe wrapper for real-time tracking
├── mlp_classifier.py        # Core: MLP wrapper used by pipeline
├── rule_based.py            # Core: Rule-based classifier used by pipeline
└── face_landmarker.task     # Model file: Required by MediaPipe Tasks API

```

---

### File Breakdown

| File Name | Component Type | Description |
| --- | --- | --- |
| **`collect_dataset.py`** | `Step 1` | Extracts features from datasets. |
| **`train_evaluate.py`** | `Step 2` | Handles training and generates all performance graphs. |
| **`model_evaluation.ipynb`** | `Step 3` | Jupyter Notebook for ground truth vs. predicted analysis. |
| **`pipeline.py`** | `Step 4` | Manages real-time webcam inference. |
| **`feature_extractor.py`** | `Core` | Computes the 9 features (shared by `collect_dataset.py` and `pipeline.py`). |
| **`face_mesh.py`** | `Core` | MediaPipe wrapper for real-time face mesh tracking. |
| **`mlp_classifier.py`** | `Core` | MLP (Multi-Layer Perceptron) wrapper used by the pipeline. |
| **`rule_based.py`** | `Core` | Rule-based classifier used by the pipeline. |
| **`face_landmarker.task`** | `Model File` | The trained model file required by the MediaPipe Tasks API. |

### 

# Once only — download images and extract features
python src/download_dataset.py
python src/classifiers/landmark_pipeline/extract_features_dataset.py

# Train all three variants (all read the same CSV)
python src/classifiers/landmark_pipeline/train_evaluate_dataset.py
python src/classifiers/landmark_pipeline/train_evaluate_dataset_v1.py
python src/classifiers/landmark_pipeline/train_evaluate_dataset_v2.py

| Script | Config | Output |
| --- | --- | --- | --- |
|t rain_evaluate_dataset.py | Baseline (64,32) | relu |	models/MLP_dataset/ |
| train_evaluate_dataset_v1.py	| V1 (128,64,32) | relu + early stop | 	models/MLP_V1_dataset/ |
| train_evaluate_dataset_v2.py	| V2 (256,128,64) | tanh + early stop |	models/MLP_V2_dataset/ |


All three use the same dataset/labeled_features_dataset.csv and random_state=42
