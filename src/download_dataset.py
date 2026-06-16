#!/usr/bin/env python3
"""
Build a balanced image dataset from the four sources used by the MLP pipeline.

Downloads (if not already present) and organises images into:

    dataset/
        Attentive/    (752 images)
        Sleepy/       (752 images)
        Distracted/   (752 images)

Sources
-------
1. Kaggle   shivampandey1233/drowsy-dataset        (Active→Attentive, Fatigue+Yawning→Sleepy)
2. Roboflow neurosense/user-attention      v1       (YOLO — face crops)
3. Roboflow distractless/distractless      v1       (YOLO — face crops)
4. Roboflow distracteddetection/distracted_detection v1  (folder — class images)

Label mapping matches collect_dataset.py exactly.

Prerequisites
-------------
    pip install kaggle roboflow opencv-python

Kaggle credentials  — one of:
  • Place kaggle.json at ~/.kaggle/kaggle.json
    (get from kaggle.com → Settings → API → Create New Token)
  • Set KAGGLE_USERNAME and KAGGLE_KEY environment variables

Roboflow credentials:
  • Set ROBOFLOW_API_KEY below (or leave empty to skip Roboflow auto-download
    and point the script at already-downloaded folders instead)

Usage
-----
    python src/download_dataset.py

After it finishes:
    git add dataset/
    git commit -m "add balanced 752-per-class image dataset (MLP source)"
"""

import os
import random
import shutil
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = ''   # paste your key here, or set as env var ROBOFLOW_API_KEY

SEED   = 42
TARGET = 752

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / 'data'       # existing local data (gitignored)
DATASET_DIR = REPO_ROOT / 'dataset'   # output — NOT gitignored, committable

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

LABEL_NAMES = {0: 'Attentive', 1: 'Sleepy', 2: 'Distracted'}

# Label maps — identical to collect_dataset.py
KAGGLE_LABEL_MAP = {
    'active subjects':  0,
    'fatigue subjects': 1,
    'yawning subjects': 1,
}

ROBOFLOW_LABEL_MAP = {
    'attentive':       0,
    'focused':         0,
    'engaged':         0,
    'awake':           0,
    'normal':          0,
    'non-distracted':  0,
    'drowsy':          1,
    'sleepy':          1,
    'tired':           1,
    'eyes closed':     1,
    'eyes_closed':     1,
    'distracted':      2,
    'inattentive':     2,
    'not_focused':     2,
    'unfocused':       2,
    'looking_away':    2,
}


# ── Credential checks ─────────────────────────────────────────────────────────
def _check_kaggle():
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists() and not (
        os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
    ):
        sys.exit(
            '\nKaggle credentials not found.\n'
            '  Option 1: place kaggle.json at ~/.kaggle/kaggle.json\n'
            '            (get from kaggle.com → Settings → API → Create New Token)\n'
            '  Option 2: set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n'
        )


def _roboflow_key():
    return ROBOFLOW_API_KEY or os.environ.get('ROBOFLOW_API_KEY', '')


# ── Download helpers ──────────────────────────────────────────────────────────
def _download_kaggle(dataset_id: str, dest: Path):
    if dest.exists() and any(dest.rglob('*.jpg')):
        print(f'  [SKIP] {dest.name} already present')
        return
    try:
        import kaggle
    except ImportError:
        sys.exit('kaggle package not installed. Run: pip install kaggle')
    _check_kaggle()
    tmp = REPO_ROOT / '_kaggle_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        print(f'  Downloading kaggle:{dataset_id} …')
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_id, path=str(tmp), unzip=True, quiet=False)
        # Move to dest
        if dest.exists():
            shutil.rmtree(dest)
        # Find extracted root (may be nested)
        candidates = sorted(tmp.rglob('Image-dataset'), key=lambda p: len(p.parts))
        src = candidates[0].parent if candidates else tmp
        shutil.move(str(src), str(dest))
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def _download_roboflow(workspace, project, version, dest: Path, fmt='yolov8'):
    key = _roboflow_key()
    if dest.exists() and any(dest.rglob('*.jpg')):
        print(f'  [SKIP] {dest.name} already present')
        return
    if not key:
        print(f'  [SKIP] ROBOFLOW_API_KEY not set — skipping {project}')
        return
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit('roboflow package not installed. Run: pip install roboflow')
    print(f'  Downloading roboflow:{workspace}/{project} v{version} …')
    if dest.exists():
        shutil.rmtree(dest)
    rf = Roboflow(api_key=key)
    rf.workspace(workspace).project(project).version(version).download(
        fmt, location=str(dest), overwrite=True
    )


# ── Image collection helpers ──────────────────────────────────────────────────
def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _collect_kaggle(kaggle_dir: Path) -> dict[int, list[Path]]:
    """Returns {label_id: [image_paths]} from folder-per-class Kaggle layout."""
    by_label: dict[int, list[Path]] = {0: [], 1: [], 2: []}
    img_root = kaggle_dir / 'Image-dataset'
    search_root = img_root if img_root.exists() else kaggle_dir
    for d in sorted(search_root.iterdir()):
        if not d.is_dir():
            continue
        key = d.name.lower()
        if key not in KAGGLE_LABEL_MAP:
            continue
        lbl = KAGGLE_LABEL_MAP[key]
        imgs = _images_in(d)
        print(f'    {d.name}/  →  {LABEL_NAMES[lbl]}  ({len(imgs)} images)')
        by_label[lbl].extend(imgs)
    return by_label


def _collect_roboflow_yolo(data_dir: Path, name: str) -> dict[int, list[Path]]:
    """Returns {label_id: [image_paths]} from a YOLO-format Roboflow dataset.
    Uses bounding-box crops to match what collect_dataset.py fed into MediaPipe.
    """
    import yaml, cv2
    by_label: dict[int, list[Path]] = {0: [], 1: [], 2: []}

    yaml_files = list(data_dir.rglob('data.yaml'))
    if not yaml_files:
        print(f'    [SKIP] data.yaml not found in {data_dir}')
        return by_label

    with open(yaml_files[0]) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get('names', [])
    id_to_label = {i: ROBOFLOW_LABEL_MAP[n.lower()]
                   for i, n in enumerate(class_names)
                   if n.lower() in ROBOFLOW_LABEL_MAP}
    print(f'    {name} classes: {class_names}')

    crop_dir = data_dir / '_crops'
    crop_dir.mkdir(exist_ok=True)
    counter = 0

    for split in ('train', 'valid', 'test'):
        imgs_dir = data_dir / split / 'images'
        lbls_dir = data_dir / split / 'labels'
        if not imgs_dir.exists():
            continue
        for img_path in sorted(imgs_dir.glob('*.*')):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lp = lbls_dir / (img_path.stem + '.txt')
            if not lp.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in lp.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid not in id_to_label:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0,     int((cx - bw / 2) * w))
                y1 = max(0,     int((cy - bh / 2) * h))
                x2 = min(w - 1, int((cx + bw / 2) * w))
                y2 = min(h - 1, int((cy + bh / 2) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                out = crop_dir / f'{counter:06d}.jpg'
                cv2.imwrite(str(out), crop)
                by_label[id_to_label[cid]].append(out)
                counter += 1

    for lbl, paths in by_label.items():
        if paths:
            print(f'    {LABEL_NAMES[lbl]}: {len(paths)} crops')
    return by_label


def _collect_roboflow_classification(data_dir: Path, name: str) -> dict[int, list[Path]]:
    """Returns {label_id: [image_paths]} from folder-per-class Roboflow classification layout."""
    by_label: dict[int, list[Path]] = {0: [], 1: [], 2: []}
    for split in ('train', 'valid', 'test'):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            key = class_dir.name.lower()
            if key not in ROBOFLOW_LABEL_MAP:
                continue
            lbl = ROBOFLOW_LABEL_MAP[key]
            imgs = _images_in(class_dir)
            by_label[lbl].extend(imgs)
    for lbl, paths in by_label.items():
        if paths:
            print(f'    {LABEL_NAMES[lbl]}: {len(paths)} images')
    return by_label


def _merge(dicts: list[dict[int, list[Path]]]) -> dict[int, list[Path]]:
    merged: dict[int, list[Path]] = {0: [], 1: [], 2: []}
    for d in dicts:
        for lbl, paths in d.items():
            merged[lbl].extend(paths)
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import cv2

    rng = random.Random(SEED)

    kaggle_dir   = DATA_DIR / 'kaggle_drowsy'
    rf_attn_dir  = DATA_DIR / 'roboflow_attention'
    rf_dless_dir = DATA_DIR / 'roboflow_distractless'
    rf_dist_dir  = DATA_DIR / 'roboflow_distracted'

    # ── Download phase ────────────────────────────────────────────────────────
    print('=== Downloading datasets ===')
    _download_kaggle('shivampandey1233/drowsy-dataset', kaggle_dir)
    _download_roboflow('neurosense',          'user-attention',       1, rf_attn_dir,  fmt='yolov8')
    _download_roboflow('distractless',        'distractless',         1, rf_dless_dir, fmt='yolov8')
    _download_roboflow('distracteddetection', 'distracted_detection', 1, rf_dist_dir,  fmt='folder')

    # ── Collect phase ─────────────────────────────────────────────────────────
    print('\n=== Collecting images ===')
    by_labels = []

    print('  [1/4] Kaggle: shivampandey1233/drowsy-dataset')
    if kaggle_dir.exists():
        by_labels.append(_collect_kaggle(kaggle_dir))
    else:
        print('    [SKIP] not found')

    print('  [2/4] Roboflow: neurosense/user-attention')
    if rf_attn_dir.exists():
        by_labels.append(_collect_roboflow_yolo(rf_attn_dir, 'user-attention'))
    else:
        print('    [SKIP] not found')

    print('  [3/4] Roboflow: distractless/distractless')
    if rf_dless_dir.exists():
        by_labels.append(_collect_roboflow_yolo(rf_dless_dir, 'distractless'))
    else:
        print('    [SKIP] not found')

    print('  [4/4] Roboflow: distracteddetection/distracted_detection')
    if rf_dist_dir.exists():
        by_labels.append(_collect_roboflow_classification(rf_dist_dir, 'distracted_detection'))
    else:
        print('    [SKIP] not found')

    merged = _merge(by_labels)

    print('\n=== Raw totals before balancing ===')
    for lbl, paths in merged.items():
        print(f'  {LABEL_NAMES[lbl]:<12} {len(paths):>5} images')

    # ── Balance & copy ────────────────────────────────────────────────────────
    print(f'\n=== Building dataset (target: {TARGET} per class) ===')
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for lbl, paths in merged.items():
        cls_name = LABEL_NAMES[lbl]
        dst_dir  = DATASET_DIR / cls_name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir()

        total = len(paths)
        if total == 0:
            print(f'  {cls_name:<12}    0 images  [WARNING: no source images found]')
            continue

        selected = rng.sample(paths, min(TARGET, total))
        note = f'sampled {len(selected)} of {total}' if total > TARGET else f'all {total} (< {TARGET})'

        for i, src in enumerate(selected):
            dst = dst_dir / f'{i:05d}{src.suffix.lower()}'
            shutil.copy2(src, dst)

        print(f'  {cls_name:<12} {len(selected):>4} images  [{note}]')

    # Clean up crop temp folders
    for rf_dir in (rf_attn_dir, rf_dless_dir):
        crop_dir = rf_dir / '_crops'
        if crop_dir.exists():
            shutil.rmtree(crop_dir)

    print(
        f'\nDataset ready at dataset/\n'
        f'Stage with:\n'
        f'  git add dataset/\n'
        f'  git commit -m "add balanced 752-per-class image dataset (MLP source)"'
    )


if __name__ == '__main__':
    main()
