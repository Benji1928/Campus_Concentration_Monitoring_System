"""
Evaluate the trained MLP on held-out test images and save confusion matrices.

Reads images from test/ (or a zip of the same structure), extracts MediaPipe
landmarks + 9 features, runs the trained MLP, then saves to the model directory:

    confusion_matrix_test.png       — raw counts
    confusion_matrix_test_norm.png  — row-normalised (recall per class)
    test_predictions.txt            — per-image filename / GT / Pred / conf

Expected folder structure (inside test/ or the zip root):
    Attentive/  →  label 0  (ATTENTIVE)
    Drowsy/     →  label 1  (SLEEPY)
    Distracted/ →  label 2  (DISTRACTED)

Usage (from project root):
    python src/classifiers/landmark_pipeline/eval_test_confusion.py
    python src/classifiers/landmark_pipeline/eval_test_confusion.py --zip test_crop.zip
    python src/classifiers/landmark_pipeline/eval_test_confusion.py --model models/MLP_V1
    python src/classifiers/landmark_pipeline/eval_test_confusion.py --model models/MLP_V2
"""

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import seaborn as sns
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from src.classifiers.landmark_pipeline.feature_extractor import FeatureExtractor, FEATURE_COLS

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_TEST_DIR  = ROOT / 'test'
DEFAULT_MODEL_DIR = ROOT / 'models' / 'MLP'

TEST_CLASS_MAP = {
    'Attentive':  0,
    'Sleepy':     1,
    'Distracted': 2,
}
LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}
COLORS      = ['#2ecc71', '#e67e22', '#e74c3c']
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}

# ── MediaPipe setup ────────────────────────────────────────────────────────────
_MODEL_TASK = Path(__file__).parent / 'face_landmarker.task'
if not _MODEL_TASK.exists():
    _URL = ('https://storage.googleapis.com/mediapipe-models/'
            'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    print('Downloading face_landmarker.task …')
    urllib.request.urlretrieve(_URL, str(_MODEL_TASK))
    print('  Done.')

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_TASK)),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
    )
)


# ── Feature extraction ─────────────────────────────────────────────────────────
def _extract_features(img_bgr):
    """Return feature vector (1-D numpy array) or None if no face detected."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    lm = [(lm.x * w, lm.y * h, lm.z) for lm in face]
    fe = FeatureExtractor(frame_w=w, frame_h=h)
    feats = fe.extract(lm)
    return np.array([feats[col] for col in FEATURE_COLS], dtype=np.float32)


# ── Collect test images ────────────────────────────────────────────────────────
def collect_images(test_dir: Path):
    """
    Walk test_dir sub-folders defined in TEST_CLASS_MAP.
    Returns list of (image_path, label_int) in deterministic sorted order.
    """
    samples = []
    for folder_name, label_id in TEST_CLASS_MAP.items():
        cls_dir = test_dir / folder_name
        if not cls_dir.exists():
            print(f'  [SKIP] {folder_name}/ not found in {test_dir}')
            continue
        paths = sorted(p for p in cls_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        print(f'  {folder_name:<12} {len(paths):>3} images  (label {label_id})')
        samples.extend((p, label_id) for p in paths)
    return samples


# ── Run predictions ────────────────────────────────────────────────────────────
def predict_all(samples, scaler, mlp):
    """
    Returns parallel lists: y_true, y_pred, y_conf, results.
    results is a list of dicts with keys: path, gt, pred, conf, face_found.
    """
    y_true, y_pred, y_conf = [], [], []
    results = []
    total = len(samples)
    for i, (img_path, gt_label) in enumerate(samples, 1):
        print(f'\r  Processing {i}/{total} …', end='', flush=True)
        img = cv2.imread(str(img_path))
        if img is None:
            results.append({'path': img_path, 'gt': gt_label,
                            'pred': None, 'conf': None, 'face_found': False})
            continue
        feat = _extract_features(img)
        if feat is None:
            results.append({'path': img_path, 'gt': gt_label,
                            'pred': None, 'conf': None, 'face_found': False})
            continue
        X_s   = scaler.transform(feat.reshape(1, -1))
        pred  = int(mlp.predict(X_s)[0])
        conf  = float(mlp.predict_proba(X_s)[0, pred])
        y_true.append(gt_label)
        y_pred.append(pred)
        y_conf.append(conf)
        results.append({'path': img_path, 'gt': gt_label,
                        'pred': pred, 'conf': conf, 'face_found': True})
    print()
    return y_true, y_pred, y_conf, results


# ── Confusion matrix plots ─────────────────────────────────────────────────────
def plot_confusion_matrices(y_true, y_pred, out_dir: Path):
    label_list = list(LABEL_NAMES.values())
    cm = np.array(
        [[sum(1 for t, p in zip(y_true, y_pred) if t == r and p == c)
          for c in range(3)]
         for r in range(3)],
        dtype=int,
    )

    # ── Raw counts ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list,
                linewidths=0.5)
    plt.title('Confusion Matrix — Test Images (counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    raw_path = out_dir / 'confusion_matrix_test.png'
    plt.savefig(raw_path, dpi=150)
    plt.close()
    print(f'  Saved {raw_path.name}')

    # ── Normalised (row = recall) ────────────────────────────────────────────────
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list,
                vmin=0, vmax=1, linewidths=0.5)
    plt.title('Normalised Confusion Matrix — Test Images (recall)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    norm_path = out_dir / 'confusion_matrix_test_norm.png'
    plt.savefig(norm_path, dpi=150)
    plt.close()
    print(f'  Saved {norm_path.name}')

    return cm, cm_norm


# ── Text report ────────────────────────────────────────────────────────────────
def save_text_report(results, y_true, y_pred, cm, out_dir: Path):
    lines = []
    lines.append('Test Image Predictions')
    lines.append('=' * 80)
    lines.append(f'{"File":<55} {"GT":<12} {"Pred":<12} {"Conf":>6}  {"Result"}')
    lines.append('-' * 80)

    per_class = {i: {'total': 0, 'pass': 0, 'fail': 0, 'no_face': 0}
                 for i in range(3)}

    for r in results:
        gt_name   = LABEL_NAMES[r['gt']]
        per_class[r['gt']]['total'] += 1
        filename = r['path'].name
        if not r['face_found']:
            lines.append(f'{filename:<55} {gt_name:<12} {"—":<12} {"—":>6}  NO FACE')
            per_class[r['gt']]['no_face'] += 1
        else:
            pred_name = LABEL_NAMES[r['pred']]
            status    = 'PASS' if r['pred'] == r['gt'] else 'FAIL'
            conf_str  = f"{r['conf']:.1%}"
            lines.append(f'{filename:<55} {gt_name:<12} {pred_name:<12} {conf_str:>6}  {status}')
            if r['pred'] == r['gt']:
                per_class[r['gt']]['pass'] += 1
            else:
                per_class[r['gt']]['fail'] += 1

    lines.append('')
    lines.append('Per-class summary')
    lines.append('-' * 50)
    for lbl, name in LABEL_NAMES.items():
        d = per_class[lbl]
        scored  = d['pass'] + d['fail']
        acc_str = f"{d['pass']/scored:.1%}" if scored > 0 else 'n/a'
        lines.append(
            f'  {name:<12}  total={d["total"]}  pass={d["pass"]}  '
            f'fail={d["fail"]}  no_face={d["no_face"]}  acc={acc_str}'
        )

    if y_true:
        overall = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        lines.append('')
        lines.append(f'OVERALL accuracy (faces only): {overall:.1%}  '
                     f'({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})')

    lines.append('')
    lines.append('Confusion matrix (rows=actual, cols=predicted):')
    header = f'{"":12}' + ''.join(f'{n:>12}' for n in LABEL_NAMES.values())
    lines.append(header)
    for r in range(3):
        row = f'{LABEL_NAMES[r]:<12}' + ''.join(f'{cm[r,c]:>12d}' for c in range(3))
        lines.append(row)

    txt_path = out_dir / 'test_predictions.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved {txt_path.name}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP on test images')
    parser.add_argument('--zip',   default=None,
                        help='Path to zip file (Attentive/, Drowsy/, Distracted/ inside)')
    parser.add_argument('--test',  default=str(DEFAULT_TEST_DIR),
                        help=f'Test image folder (default: {DEFAULT_TEST_DIR})')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_DIR),
                        help=f'Model directory (default: {DEFAULT_MODEL_DIR})')
    args = parser.parse_args()

    model_dir = Path(args.model)

    # ── Load model ─────────────────────────────────────────────────────────────
    pkl_mlp    = model_dir / 'mlp_model.pkl'
    pkl_scaler = model_dir / 'scaler.pkl'
    if not pkl_mlp.exists() or not pkl_scaler.exists():
        sys.exit(
            f'\nModel files not found in {model_dir}\n'
            'Run train_evaluate_dataset.py first.'
        )
    mlp    = joblib.load(pkl_mlp)
    scaler = joblib.load(pkl_scaler)
    print(f'Loaded model from {model_dir.name}/')

    # ── Resolve test directory ─────────────────────────────────────────────────
    tmp_dir = None
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            sys.exit(f'\nZip file not found: {zip_path}')
        tmp_dir  = tempfile.mkdtemp(prefix='eval_test_')
        test_dir = Path(tmp_dir)
        print(f'Extracting {zip_path.name} …')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(test_dir)
        # If zip has a single top-level folder, descend into it
        contents = list(test_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            test_dir = contents[0]
    else:
        test_dir = Path(args.test)
        if not test_dir.exists():
            sys.exit(f'\nTest directory not found: {test_dir}')

    print(f'\nScanning {test_dir} …')
    samples = collect_images(test_dir)
    if not samples:
        sys.exit('\nNo images found — check folder names.')

    print(f'\nRunning MLP on {len(samples)} images …')
    y_true, y_pred, y_conf, results = predict_all(samples, scaler, mlp)

    no_face = sum(1 for r in results if not r['face_found'])
    print(f'  {len(y_true)} predictions made  |  {no_face} images skipped (no face)')

    if not y_true:
        sys.exit('\nNo faces detected in any image — cannot generate confusion matrix.')

    print('\nSaving outputs to', model_dir)
    cm, cm_norm = plot_confusion_matrices(y_true, y_pred, model_dir)
    save_text_report(results, y_true, y_pred, cm, model_dir)

    overall = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f'\nOverall accuracy: {overall:.1%}  ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)} faces)')

    # clean up temp extract
    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
