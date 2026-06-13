# YOLO Pipeline

One of two pipelines in this project. Classifies student engagement states — **Attentive**, **Distracted**, or **Drowsy** — using a two-stage deep learning approach: YOLOv8 face detection followed by a MobileNetV3 CNN classifier trained via transfer learning.

---

## How It Works

```
TRAINING TIME
Raw images → face_detection.ipynb → best.pt (face detector)
                                          │
Raw engagement dataset ──────────────────┘
       │   crop_face.ipynb uses best.pt to crop faces
       ▼
Cropped face dataset → CNN_training.ipynb → best_driver_state_model.pth

INFERENCE TIME  (runtime pipeline not yet built)
Webcam frame → YOLOv8 (best.pt) → crop face → MobileNetV3 → ATTENTIVE / DISTRACTED / DROWSY
```

---

## File Reference

| File | Purpose | Key Details |
|---|---|---|
| `face_detection.ipynb` | Trains YOLOv8n face detector on Colab | 299 images, 80/10/10 split, 50 epochs, 1 class: `face`, outputs `best.pt` |
| `crop_face.ipynb` | Crops faces from raw engagement dataset using trained detector | Processes 4,341 images across Attentive / Distracted / Drowsy, outputs cleaned zip (165 MB) |
| `CNN_training.ipynb` | Fine-tunes MobileNetV3-Large on cropped face images | Transfer learning, 15 epochs, early stopping (patience=3), 85/15 stratified split, outputs `best_driver_state_model.pth` |
| `README.md` | Documentation for this pipeline | — |

---

## Training Order

**Step 1 — Train the face detector**

Run `face_detection.ipynb` on Google Colab.
Outputs `best.pt` — a YOLOv8 model that detects faces in images.

**Step 2 — Crop faces from the engagement dataset**

Run `crop_face.ipynb` on Kaggle using `best.pt`.
Outputs a cleaned dataset of tightly cropped face images organised into class folders:
```
Attentive/
Distracted/
Drowsy/
```

**Step 3 — Train the engagement classifier**

Run `CNN_training.ipynb` on Kaggle using the cropped dataset.
Outputs `best_driver_state_model.pth` — the trained MobileNetV3 model.

---

## Model Details

### Face Detector — YOLOv8n (`best.pt`)
| Property | Value |
|---|---|
| Architecture | YOLOv8 nano |
| Classes | 1 (`face`) |
| Training images | 239 train / 30 val / 30 test |
| Epochs | 50 |
| Platform | Google Colab |

### Engagement Classifier — MobileNetV3-Large (`best_driver_state_model.pth`)
| Property | Value |
|---|---|
| Architecture | MobileNetV3-Large (pretrained ImageNet) |
| Classes | 3 (Attentive, Distracted, Drowsy) |
| Input size | 224 × 224 |
| Train / Val split | 85% / 15% (stratified) |
| Epochs | 15 (early stopping, patience=3) |
| Platform | Kaggle (GPU) |

---

## Checklist

- [x] Train face detector (`face_detection.ipynb`)
- [x] Crop face dataset (`crop_face.ipynb`)
- [x] Train CNN classifier (`CNN_training.ipynb`)
- [ ] Build runtime `pipeline.py` to run inference from webcam
