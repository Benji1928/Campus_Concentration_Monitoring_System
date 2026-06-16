"""
Extract landmark features from the committed dataset/ images and save to
dataset/labeled_features_dataset.csv.

Run this once after download_dataset.py. The output CSV is committed to git
so teammates can skip this step and go straight to train_evaluate_dataset.py.

Pipeline
--------
1. Read images from dataset/Attentive/, dataset/Sleepy/, dataset/Distracted/
   in sorted filename order (deterministic across all machines)
2. Pass each image through MediaPipe FaceLandmarker (478-point Tasks API)
   Discard images where no face is detected
3. Extract 9 landmark features via FeatureExtractor
4. Balance to minority class count using seed=42
5. Save to dataset/labeled_features_dataset.csv

Usage (from project root):
    python src/classifiers/landmark_pipeline/extract_features_dataset.py
"""

import csv
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from src.classifiers.landmark_pipeline.feature_extractor import FeatureExtractor, FEATURE_COLS

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42

DATASET_DIR = ROOT / 'dataset'
OUTPUT_CSV  = DATASET_DIR / 'labeled_features_dataset.csv'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

CLASS_MAP = {
    'Attentive':  0,
    'Sleepy':     1,
    'Distracted': 2,
}
LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}

# ── MediaPipe setup ────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'
if not _MODEL_PATH.exists():
    _URL = ('https://storage.googleapis.com/mediapipe-models/'
            'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    print('Downloading face_landmarker.task …')
    urllib.request.urlretrieve(_URL, str(_MODEL_PATH))
    print('  Done.')

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
    )
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_landmarks(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    return [(lm.x * w, lm.y * h, lm.z) for lm in face]


# ── Extraction ─────────────────────────────────────────────────────────────────
def extract() -> dict[int, list[dict]]:
    """
    Reads dataset/ in sorted filename order (deterministic).
    Returns {label_id: [feature_dict, ...]} for images where MediaPipe found a face.
    """
    if not DATASET_DIR.exists():
        sys.exit(
            f'\ndataset/ not found at {DATASET_DIR}\n'
            'Run python src/download_dataset.py first.'
        )

    by_label: dict[int, list[dict]] = {0: [], 1: [], 2: []}

    for cls_name, label_id in CLASS_MAP.items():
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.exists():
            print(f'  [SKIP] {cls_name}/ not found in dataset/')
            continue

        images = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        detected, skipped = 0, 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            lm = _get_landmarks(img)
            if lm is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            fe = FeatureExtractor(frame_w=w, frame_h=h)
            features = fe.extract(lm)
            row = {col: round(features[col], 4) for col in FEATURE_COLS}
            row['label'] = label_id
            by_label[label_id].append(row)
            detected += 1

        print(f'  {cls_name:<12}  {detected:>4} detected   {skipped:>3} skipped   (of {len(images)} images)')

    return by_label


# ── Balance ────────────────────────────────────────────────────────────────────
def balance(by_label: dict[int, list[dict]]) -> list[dict]:
    """Undersample to the minority class using seed=42, then shuffle with seed=42."""
    rng = random.Random(SEED)
    min_count = min(len(rows) for rows in by_label.values())
    print(f'\nBalancing to {min_count} rows per class (seed={SEED})')
    balanced = []
    for label_id, rows in by_label.items():
        selected = rng.sample(rows, min_count)
        balanced.extend(selected)
        print(f'  {LABEL_NAMES[label_id]:<12}  {min_count} rows  (from {len(rows)} detected)')
    rng.shuffle(balanced)
    return balanced


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=== Step 1: Feature extraction from dataset/ ===')
    by_label = extract_features()

    print(f'\nRaw detected totals:')
    for label_id, rows in by_label.items():
        print(f'  {LABEL_NAMES[label_id]:<12}  {len(rows)}')

    print('\n=== Step 2: Balancing ===')
    balanced = balance(by_label)

    print('\n=== Step 3: Saving CSV ===')
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS + ['label'])
        writer.writeheader()
        writer.writerows(balanced)
    print(f'  Saved {len(balanced)} rows → dataset/labeled_features_dataset.csv')
    print('\nNext step:')
    print('  python src/classifiers/landmark_pipeline/train_evaluate_dataset.py')


def extract_features():
    return extract()


if __name__ == '__main__':
    main()
