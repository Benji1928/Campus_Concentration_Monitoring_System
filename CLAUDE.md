# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time CV system with three entry points:
1. **Attention pipeline** — MediaPipe face landmarks → EAR/MAR/head-pose features → rule-based or MLP classifier
2. **Face counter** — YOLOv8 detection → Tkinter dashboard with live stats
3. **Unified dashboard** — DearPyGui app: YOLOv8 face detection → user-selectable classifier (rule-based / MLP / MobileNetV3 / EfficientNetV2)

## Commands

```bash
# Setup (Python 3.11 required — mediapipe has strict version requirements)
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
pip install ultralytics pillow  # face_counter.py deps (not in requirements.txt)
pip install dearpygui           # src/dashboard.py only

# Attention pipeline (rule-based only)
python src/pipeline.py

# Attention pipeline (rule-based + MLP side-by-side — requires trained model)
python src/pipeline.py --mlp

# Different camera index
python src/pipeline.py --camera 1

# Face counter dashboard
python src/face_counter.py
python src/face_counter.py --model models/face_detection.pt --conf 0.75 --cam 0

# Train MLP (requires data/labeled_features.csv)
python src/training/train.py

# Unified dashboard (DearPyGui)
python src/dashboard.py
```

No test suite exists.

## Architecture

### Data flow — attention pipeline

```
FaceMesh.process(frame)          # MediaPipe → 468 (x,y,z) landmarks or None
  → FeatureExtractor.extract()   # → dict with 9 features (FEATURE_COLS)
  → RuleBasedClassifier.predict()     # threshold rules, no training needed
  → MLPAttentionClassifier.predict()  # optional, requires trained pkl
```

### Data flow — unified dashboard pipeline (`src/inference_pipeline.py`)

```
YOLOFaceDetector.detect(frame)       # YOLOv8n → list[FaceDetection] (bbox + conf)
  → FaceMesh.process(frame)          # only when landmark classifier selected
  → FeatureExtractor.extract()       # → 9-feature dict
  → active classifier.predict()      # → ClassifierResult (label, label_int, probs, conf)
```

Classifier swapped at runtime via `InferencePipeline.set_classifier()`. To add a new classifier: subclass `BaseAttentionClassifier` in `src/classifiers/base.py`, add entry to `build_classifier_registry()` in `src/inference_pipeline.py`.

### Feature set (`FEATURE_COLS` order matters for MLP input)

`ear_left`, `ear_right`, `ear_avg`, `mar`, `pitch`, `yaw`, `roll`, `perclos`, `blink_rate`

- **EAR** — 6-point eye aspect ratio (indices: `RIGHT_EYE`, `LEFT_EYE` in `feature_extractor.py`)
- **MAR** — same formula applied to mouth landmarks (`MOUTH`)
- **PERCLOS** — rolling fraction of frames with `ear_avg < 0.22` (90-frame window)
- **Head pose** — `cv2.solvePnP` on 6 landmarks (`HEAD_POSE_LM`) against a fixed 3-D face model

### Models directory

| File | Used by |
|------|---------|
| `face_landmarker.task` | `FaceMesh` — auto-downloaded from Google on first run |
| `face_detection.pt` | `YOLOFaceDetector`, `face_counter.py` — YOLOv8n face detection |
| `mlp_model.pkl` + `scaler.pkl` | `MLPAttentionClassifier.load()` |
| `best_driver_state_model.pth` | `MobileNetV3Classifier` — MobileNetV3 Large, 3-class, 224×224 |
| `best_efficientnetv2_driver_model.pth` | `EfficientNetV2Classifier` — EfficientNetV2-S, 3-class, 224×224 |

### Classifiers

- `BaseAttentionClassifier` (`src/classifiers/base.py`) — ABC; all classifiers return `ClassifierResult(label, label_int, probabilities, confidence)`
- `RuleBasedClassifier` — pure thresholds (no training); `SLEEPY` takes priority over `DISTRACTED`
- `MLPAttentionClassifier` — sklearn `MLPClassifier(hidden_layer_sizes=(64,32))` with `StandardScaler`; trained via `src/training/train.py` from `data/labeled_features.csv`
- `MobileNetV3Classifier` / `EfficientNetV2Classifier` (`src/classifiers/dl_classifier.py`) — load `.pth` state dict; `needs_landmarks=False`; input: 224×224 BGR crop

### `src/dashboard.py`

Full DearPyGui dashboard. Worker thread → `queue.Queue(maxsize=2)` → main render loop (`while dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()`). Do NOT use `dpg.start_dearpygui()` — blocks and prevents background thread integration.

## Key constraints

- `FeatureExtractor` is stateful (PERCLOS ring buffer, blink timestamps) — one instance per camera session; call `reset()` on reconnect.
- MLP input order is fixed by `FEATURE_COLS`; adding/removing features requires retraining.
- `face_counter.py` uses `ultralytics` and `pillow` which are absent from `requirements.txt`.
- `src/data/collector.py` does not exist; data must be collected manually and saved to `data/labeled_features.csv` with columns matching `FEATURE_COLS` + `label` (int 0/1/2).
- Rule-based thresholds (`EAR_SLEEPY`, `PERCLOS_SLEEPY`, `MAR_YAWN`, `YAW_DISTRACTED`, `PITCH_DISTRACTED`) live at the top of `src/classifiers/rule_based.py` — primary tuning surface without retraining.
- `.pth` checkpoint format: `_load_weights()` in `dl_classifier.py` probes keys `model_state_dict` → `state_dict` → `model` → flat state dict → full model save. If a new model fails to load, check its save format.
- Dashboard texture lag: `_to_rgba()` converts 640×480 frame to ~1.2M-element Python float list each frame — known bottleneck. Fix: replace `.tolist()` with `numpy` bytes buffer / `memoryview`.
