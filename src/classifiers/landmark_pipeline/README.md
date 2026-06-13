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

## Installation

```powershell
pip install "numpy<2.0"
pip install mediapipe opencv-python scikit-learn pandas joblib
```

---

## Step 1 — Collect Training Data

Run the data collector and hold keys to label frames in real time.

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

## Step 3 — Run the Pipeline

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
