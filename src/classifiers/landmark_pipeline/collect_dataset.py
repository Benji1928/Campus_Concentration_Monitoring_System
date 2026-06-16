"""
Offline feature collector — reads downloaded Kaggle and Roboflow datasets,
extracts landmark features using the existing FeatureExtractor, and writes
labeled_features.csv to data/.

Usage (from project root):
    python src/classifiers/landmark_pipeline/collect_from_datasets.py

Datasets expected at:
    data/kaggle_drowsy/          — folder-per-class layout (shivampandey1233/drowsy-dataset)
    data/roboflow_attention/     — YOLO format (neurosense/user-attention v1)
    data/roboflow_distractless/  — YOLO format (distractless/distractless v1)
    data/roboflow_distracted/    — YOLO format (distracteddetection/distracted_detection v1)

Set ROBOFLOW_API_KEY below to enable auto-download of Roboflow datasets.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import csv
import urllib.request
import yaml
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from src.classifiers.landmark_pipeline.feature_extractor import (
    FeatureExtractor, FEATURE_COLS
)

# ── Optional: set your Roboflow API key to enable auto-download ───────────────
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')

# ── Download FaceLandmarker model if missing ──────────────────────────────────
_MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'
if not _MODEL_PATH.exists():
    _URL = ('https://storage.googleapis.com/mediapipe-models/'
            'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    print('Downloading face_landmarker.task ...')
    urllib.request.urlretrieve(_URL, str(_MODEL_PATH))
    print('  Done.')

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
    )
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_NB_DATA   = Path(__file__).parent / 'data'
_ROOT_DATA = ROOT / 'data'

def _find(name):
    nb = _NB_DATA / name
    rt = _ROOT_DATA / name
    return nb if nb.exists() else rt

KAGGLE_DIR              = _find('kaggle_drowsy')
ROBOFLOW_DIR              = _find('roboflow_attention')
ROBOFLOW_DISTRACTLESS_DIR = _find('roboflow_distractless')
ROBOFLOW_DISTRACTED_DIR   = _find('roboflow_distracted')
OUTPUT_CSV              = _ROOT_DATA / 'labeled_features.csv'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ── Label maps ────────────────────────────────────────────────────────────────
KAGGLE_LABEL_MAP = {
    'active subjects':  0,   # attentive
    'fatigue subjects': 1,   # sleepy
    'yawning subjects': 1,   # sleepy (yawning -> drowsy)
}

ROBOFLOW_LABEL_MAP = {
    'attentive':    0,
    'focused':      0,
    'engaged':      0,
    'awake':        0,
    'normal':           0,   # distractless + user-attention
    'non-distracted':   0,   # distracted_detection dataset
    'drowsy':       1,
    'sleepy':       1,
    'tired':        1,
    'eyes closed':  1,   # user-attention
    'eyes_closed':  1,
    'distracted':   2,
    'inattentive':  2,
    'not_focused':  2,
    'unfocused':    2,
    'looking_away': 2,
}

LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}


# ── Roboflow auto-download ─────────────────────────────────────────────────────
def _roboflow_download(workspace, project, version, dest_dir, fmt='yolov8'):
    """Downloads a Roboflow dataset if not already present."""
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.rglob('*.jpg')):
        print(f'  [SKIP] {dest_dir.name} already downloaded')
        return True
    if not ROBOFLOW_API_KEY:
        print(f'  [SKIP] ROBOFLOW_API_KEY not set — skipping download of {project}')
        return False
    try:
        import shutil
        from roboflow import Roboflow
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        rf.workspace(workspace).project(project).version(version).download(
            fmt, location=str(dest_dir), overwrite=True
        )
        return True
    except Exception as e:
        print(f'  [ERROR] Roboflow download failed: {e}')
        return False


# ── MediaPipe helper ──────────────────────────────────────────────────────────
def get_landmarks(img_bgr):
    """Returns list of (x_px, y_px, z_norm) or None if no face found."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]   # 478 landmarks; first 468 match FaceMesh
    return [(lm.x * w, lm.y * h, lm.z) for lm in face]


def extract_row(img_bgr, label_id):
    """Returns a CSV-ready dict or None if face not detected."""
    lm = get_landmarks(img_bgr)
    if lm is None:
        return None
    h, w = img_bgr.shape[:2]
    fe = FeatureExtractor(frame_w=w, frame_h=h)
    features = fe.extract(lm)
    row = {col: round(features[col], 4) for col in FEATURE_COLS}
    row['label'] = label_id
    return row


# ── Kaggle dataset (folder-per-class) ────────────────────────────────────────
def process_kaggle():
    rows, counts, skipped = [], {v: 0 for v in LABEL_NAMES.values()}, 0
    if not KAGGLE_DIR.exists():
        print(f'  [SKIP] {KAGGLE_DIR} not found')
        return rows
    for d in sorted(KAGGLE_DIR.rglob('*')):
        if not d.is_dir():
            continue
        key = d.name.lower()
        if key not in KAGGLE_LABEL_MAP:
            continue
        lbl_id = KAGGLE_LABEL_MAP[key]
        images = [p for p in d.glob('*.*') if p.suffix.lower() in IMG_EXTS]
        print(f'  {d.name}/  ->  {LABEL_NAMES[lbl_id]}  ({len(images)} images)')
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            row = extract_row(img, lbl_id)
            if row is None:
                skipped += 1
                continue
            rows.append(row)
            counts[LABEL_NAMES[lbl_id]] += 1
    print(f'  Kaggle totals: {counts}  skipped: {skipped}')
    return rows


# ── Roboflow YOLO dataset (generic) ──────────────────────────────────────────
def process_roboflow(data_dir, dataset_name='Roboflow'):
    """Processes any Roboflow dataset in YOLO format from the given directory."""
    rows, counts, skipped = [], {v: 0 for v in LABEL_NAMES.values()}, 0
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f'  [SKIP] {data_dir} not found')
        return rows

    yaml_files = list(data_dir.rglob('data.yaml'))
    if not yaml_files:
        print(f'  [SKIP] data.yaml not found in {data_dir}')
        return rows

    with open(yaml_files[0]) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get('names', [])
    print(f'  {dataset_name} classes: {class_names}')

    id_to_label = {i: ROBOFLOW_LABEL_MAP[n.lower()]
                   for i, n in enumerate(class_names)
                   if n.lower() in ROBOFLOW_LABEL_MAP}
    if not id_to_label:
        print(f'  [SKIP] No class names matched ROBOFLOW_LABEL_MAP')
        print(f'         Known names: {list(ROBOFLOW_LABEL_MAP.keys())}')
        return rows

    for split in ['train', 'valid', 'test']:
        imgs_dir = data_dir / split / 'images'
        lbls_dir = data_dir / split / 'labels'
        if not imgs_dir.exists():
            continue
        images = [p for p in imgs_dir.glob('*.*') if p.suffix.lower() in IMG_EXTS]
        print(f'  {split}/  ({len(images)} images)', flush=True)
        for i, img_path in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f'    {i + 1}/{len(images)}  collected: {counts}', flush=True)
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            lp = lbls_dir / (img_path.stem + '.txt')
            if not lp.exists():
                skipped += 1
                continue
            for line in lp.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid not in id_to_label:
                    continue
                label_name = LABEL_NAMES[id_to_label[cid]]
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0,     int((cx - bw / 2) * w))
                y1 = max(0,     int((cy - bh / 2) * h))
                x2 = min(w - 1, int((cx + bw / 2) * w))
                y2 = min(h - 1, int((cy + bh / 2) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    skipped += 1
                    continue
                row = extract_row(crop, id_to_label[cid])
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)
                counts[label_name] += 1

    print(f'  {dataset_name} totals: {counts}  skipped: {skipped}')
    return rows


# ── Roboflow classification dataset (folder-per-class inside splits) ──────────
def process_roboflow_classification(data_dir, dataset_name='Roboflow'):
    """Processes a Roboflow classification dataset (folder format).
    Structure: data_dir/{train,valid,test}/ClassName/image.jpg
    """
    rows, counts, skipped = [], {v: 0 for v in LABEL_NAMES.values()}, 0
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f'  [SKIP] {data_dir} not found')
        return rows

    for split in ['train', 'valid', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f'  {split}/  classes: {[d.name for d in class_dirs]}')
        for class_dir in class_dirs:
            key = class_dir.name.lower()
            if key not in ROBOFLOW_LABEL_MAP:
                print(f'    [SKIP] unmapped class: {class_dir.name}')
                continue
            lbl_id = ROBOFLOW_LABEL_MAP[key]
            images = [p for p in class_dir.glob('*.*') if p.suffix.lower() in IMG_EXTS]
            for i, img_path in enumerate(images):
                if (i + 1) % 100 == 0:
                    print(f'    {class_dir.name}: {i + 1}/{len(images)}', flush=True)
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                row = extract_row(img, lbl_id)
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)
                counts[LABEL_NAMES[lbl_id]] += 1

    print(f'  {dataset_name} totals: {counts}  skipped: {skipped}')
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Auto-download Roboflow datasets if API key is set
    print('=== Checking Roboflow datasets ===')
    _roboflow_download('neurosense', 'user-attention', 1,
                       _ROOT_DATA / 'roboflow_attention')
    _roboflow_download('distractless', 'distractless', 1,
                       _ROOT_DATA / 'roboflow_distractless')
    _roboflow_download('distracteddetection', 'distracted_detection', 1,
                       _ROOT_DATA / 'roboflow_distracted', fmt='folder')

    print('\n=== Kaggle dataset ===')
    rows_kaggle = process_kaggle()

    print('\n=== Roboflow: user-attention ===')
    rows_rf1 = process_roboflow(ROBOFLOW_DIR, 'user-attention')

    print('\n=== Roboflow: distractless ===')
    rows_rf2 = process_roboflow(ROBOFLOW_DISTRACTLESS_DIR, 'distractless')

    print('\n=== Roboflow: distracted_detection ===')
    rows_rf3 = process_roboflow_classification(ROBOFLOW_DISTRACTED_DIR, 'distracted_detection')

    all_rows = rows_kaggle + rows_rf1 + rows_rf2 + rows_rf3
    if not all_rows:
        print('\nNo rows collected. Check that datasets are downloaded first.')
        return

    # Undersample majority classes to match the minority class count
    import random
    by_label = {}
    for r in all_rows:
        by_label.setdefault(r['label'], []).append(r)
    min_count = min(len(v) for v in by_label.values())
    print(f'\nBalancing: keeping {min_count} rows per class')
    balanced = []
    for _, rows in by_label.items():
        random.shuffle(rows)
        balanced.extend(rows[:min_count])
    random.shuffle(balanced)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS + ['label'])
        writer.writeheader()
        writer.writerows(balanced)

    print(f'Saved {len(balanced)} rows -> {OUTPUT_CSV}')
    for name, n in sorted({LABEL_NAMES[l]: len(r) for l, r in by_label.items()}.items()):
        print(f'  {name}: {min_count} (from {n})')


if __name__ == '__main__':
    main()
