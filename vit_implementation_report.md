# Vision Transformer Models — Implementation Report

## Overview

Two transformer-based classifiers were implemented as alternatives to the CNN baseline for student concentration state classification. Both models target a 3-class problem: **Attentive**, **Sleepy**, **Distracted**.

Models were added in commit `5143bda` and are marked optional for inclusion in the final project.

---

## Models

### 1. DeiT-Tiny (`deit_tiny_patch16_224`)

**Data-efficient Image Transformer (DeiT)**, developed by Facebook AI. The Tiny variant uses:

- Patch size: 16×16 pixels
- Input resolution: 224×224
- Architecture: pure Vision Transformer (ViT) encoder with knowledge distillation pre-training on ImageNet

**Classification Head** (custom, replaces default head):

```
in_features (192) → Linear(192, 128) → ReLU → Dropout(0.2) → Linear(128, 3)
```

**Training Configuration:**

| Parameter | Value |
|---|---|
| Backbone | `deit_tiny_patch16_224` (pretrained, ImageNet) |
| Input size | 224 × 224 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR (η_min = 1e-6) |
| Max epochs | 15 |
| Early stopping patience | 5 |
| Min delta (early stop) | 0.001 |
| Dropout | 0.2 |
| Label smoothing | 0.0 |
| Class weights | Optional (disabled by default) |
| Val split | 20% stratified |

**Output directory:** `models/DeiT_Tiny/`

---

### 2. MobileViT-XXS (`mobilevit_xxs`)

**MobileViT** is a lightweight hybrid architecture combining MobileNet-style convolutions with transformer blocks. The XXS (extra-extra-small) variant targets mobile/edge deployment.

**Classification Head** (custom):

```
num_features → AdaptiveAvgPool2d(1) → Flatten → Linear(num_features, 3)
```

No hidden layer — direct linear projection to 3 classes (leaner than DeiT head).

**Training Configuration:**

| Parameter | Value |
|---|---|
| Backbone | `mobilevit_xxs` (pretrained, ImageNet) |
| Input size | 256 × 256 |
| Batch size | 32 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR (η_min = 1e-6) |
| Max epochs | 15 |
| Early stopping patience | 5 |
| Min delta (early stop) | 0.001 |
| Dropout | 0.1 |
| Label smoothing | 0.0 |
| Class weights | Optional (disabled by default) |
| Val split | 20% stratified |

**Output directory:** `models/MobileViT_XXS/`

---

## Dataset

- **Source:** `data/face_dataset/` (cropped face images, folder-per-class)
- **Classes:** Attentive (0), Sleepy (1), Distracted (2)
- **Approximate class distribution:**

| Class | Count |
|---|---|
| Attentive | 1,179 |
| Sleepy | 1,025 |
| Distracted | 790 |
| **Total** | **~2,994** |

- Split is stratified to preserve class ratios across train/val.
- Pre-split directories (`train_dir`, `val_dir`) are also supported if provided.

---

## Data Augmentation

Applied to training set only; validation uses clean resize + normalize.

| Transform | Parameter |
|---|---|
| Resize | Model-specific (224 or 256) |
| RandomHorizontalFlip | p = 0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness=0.2, contrast=0.2 |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet) |

---

## Training Pipeline

Both scripts share the same training loop structure:

1. Load pretrained backbone via `timm.create_model(..., num_classes=0)`
2. Replace `model.head` with custom classification head
3. Train full model (no layer freezing — all weights updated)
4. Track train/val loss and accuracy per epoch
5. Save best checkpoint on val loss improvement (`best_model.pt`)
6. Early stopping if no improvement for `patience` consecutive epochs
7. On completion: reload best checkpoint → generate confusion matrix + classification report

---

## Outputs (per model)

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint by val loss |
| `training_curves.png` | Loss and accuracy plots (train vs val) |
| `confusion_matrix.png` | Val set confusion matrix (seaborn heatmap) |
| `training_stats.txt` | Hyperparameters + classification report |

---

## Configuration System

Hyperparameters are managed via Python dataclasses in `transformer_configs.py`:

- `DeiTTinyConfig` — all DeiT-Tiny settings
- `MobileViTXXSConfig` — all MobileViT-XXS settings

Both support optional class-weight loss (`use_class_weights=True`) computed as:

```
weight_c = total_samples / (num_classes × count_c)
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `timm` | Pretrained ViT backbones |
| `torch` / `torchvision` | Training framework, data loading |
| `scikit-learn` | Stratified split, confusion matrix, classification report |
| `matplotlib` / `seaborn` | Training curves and confusion matrix plots |
| `tqdm` | Epoch progress bars |

Training was conducted on Google Colab (CUDA 12.8, Python 3.12) as documented in `Vit_training.ipynb`.

---

## Files Added

```
src/classifiers/classification_pipeline/
├── Transformer_training_deit.py        # DeiT-Tiny training script
├── Transformer_training_mobilevit.py   # MobileViT-XXS training script
├── transformer_configs.py              # Hyperparameter dataclasses
└── Vit_training.ipynb                  # Colab training notebook
```
