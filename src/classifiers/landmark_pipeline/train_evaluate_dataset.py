"""
Train and evaluate the MLP classifier from dataset/labeled_features_dataset.csv.

Requires extract_features_dataset.py to have been run first (or the CSV to be
present from git — it is committed alongside the dataset/ images).

Pipeline
--------
1. Load dataset/labeled_features_dataset.csv
2. 80/20 stratified train/test split  (random_state=42)
3. StandardScaler normalisation
4. MLPClassifier training on the 80% split only (random_state=42)
5. Evaluation #1 — the held-out 20% dataset split: confusion matrix,
   precision-recall, confidence curves
6. Evaluation #2 — test_new_crops/ (unseen, real-world cropped images):
   MediaPipe landmark extraction per image + confusion matrix
7. Save model to models/MLP_dataset/

Because the CSV is committed to git and all seeds are fixed at 42, every team
member running this script produces the exact same split, model, and metrics.

Usage (from project root):
    python src/classifiers/landmark_pipeline/train_evaluate_dataset.py

Outputs:
    dataset/labeled_features_dataset.csv   (input — committed to git)
    models/MLP_dataset/
        mlp_model.pkl
        scaler.pkl
        class_distribution.png
        feature_distributions.png
        loss_curve.png
        confusion_matrix.png                    (dataset 20% test split)
        confusion_matrix_norm.png                (dataset 20% test split)
        precision_recall_curve.png               (dataset 20% test split)
        confidence_curves.png                    (dataset 20% test split)
        confusion_matrix_test_new_crops.png      (test_new_crops/ images)
        confusion_matrix_test_new_crops_norm.png (test_new_crops/ images)
        test_new_crops_predictions.txt           (test_new_crops/ images)
        training_stats.txt
        curve_statistics.txt
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import seaborn as sns
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_curve, average_precision_score,
)

from src.classifiers.landmark_pipeline.feature_extractor import FeatureExtractor, FEATURE_COLS

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42

INPUT_CSV  = ROOT / 'dataset' / 'labeled_features_dataset.csv'
MODELS_DIR = ROOT / 'models' / 'MLP_dataset'

LABEL_NAMES = {0: 'ATTENTIVE', 1: 'SLEEPY', 2: 'DISTRACTED'}
COLORS      = {0: '#2ecc71',   1: '#e67e22', 2: '#e74c3c'}

STAT_THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

# ── External test set (unseen real-world crops, not part of the CSV split) ────
TEST_NEW_CROPS_DIR = ROOT / 'test_new_crops'
TEST_CLASS_MAP = {'Attentive': 0, 'Sleepy': 1, 'Distracted': 2}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}

_landmarker = None  # lazily created — only needed if test_new_crops/ exists


def _get_landmarker():
    """Lazily create the MediaPipe FaceLandmarker (avoids the download/init
    cost when test_new_crops/ isn't present)."""
    global _landmarker
    if _landmarker is None:
        model_path = Path(__file__).parent / 'face_landmarker.task'
        if not model_path.exists():
            url = ('https://storage.googleapis.com/mediapipe-models/'
                    'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
            print('Downloading face_landmarker.task …')
            urllib.request.urlretrieve(url, str(model_path))
            print('  Done.')
        _landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=1,
            )
        )
    return _landmarker


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(filename):
    path = MODELS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {filename}')


def _section(f, title, char='='):
    f.write(f'\n{title}\n{char * len(title)}\n')


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_class_distribution(df):
    counts = df['label'].map(LABEL_NAMES).value_counts()
    name_to_id = {v: k for k, v in LABEL_NAMES.items()}
    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values,
                   color=[COLORS[name_to_id[n]] for n in counts.index])
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 5, str(val),
                 ha='center', fontweight='bold')
    plt.title('Class Distribution')
    plt.xlabel('Engagement State')
    plt.ylabel('Samples')
    plt.tight_layout()
    _save('class_distribution.png')


def plot_feature_distributions(df):
    key_features = ['ear_avg', 'mar', 'pitch', 'yaw', 'perclos', 'blink_rate']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flatten(), key_features):
        for lbl, name in LABEL_NAMES.items():
            ax.hist(df[df['label'] == lbl][feat], bins=30,
                    alpha=0.6, label=name, color=COLORS[lbl])
        ax.set_title(feat)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
    plt.suptitle('Feature Distributions per Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('feature_distributions.png')


def plot_loss_curve(mlp):
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, color='steelblue', linewidth=2)
    plt.title('MLP Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save('loss_curve.png')


def plot_confusion_matrix(y_test, y_pred, tag='', title_suffix=''):
    """Plots + saves confusion_matrix{tag}.png / confusion_matrix{tag}_norm.png.

    tag=''                   → the held-out 20% dataset split (default, original filenames)
    tag='_test_new_crops'    → the external test_new_crops/ evaluation
    """
    label_list = list(LABEL_NAMES.values())
    cm = confusion_matrix(y_test, y_pred, labels=list(LABEL_NAMES.keys()))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list)
    plt.title(f'Confusion Matrix{title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save(f'confusion_matrix{tag}.png')

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list, vmin=0, vmax=1)
    plt.title(f'Normalised Confusion Matrix{title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save(f'confusion_matrix{tag}_norm.png')

    return cm


# ── External test (test_new_crops/) ────────────────────────────────────────────
def _extract_features_from_image(img_bgr):
    """Returns a feature vector (1-D float32 array) or None if no face detected."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    lm = [(l.x * w, l.y * h, l.z) for l in face]
    fe = FeatureExtractor(frame_w=w, frame_h=h)
    feats = fe.extract(lm)
    return np.array([feats[col] for col in FEATURE_COLS], dtype=np.float32)


def collect_test_new_crops():
    """Walks TEST_NEW_CROPS_DIR/{Attentive,Sleepy,Distracted}/, returns sorted
    list of (image_path, label_int)."""
    samples = []
    for folder_name, label_id in TEST_CLASS_MAP.items():
        cls_dir = TEST_NEW_CROPS_DIR / folder_name
        if not cls_dir.exists():
            print(f'  [SKIP] {folder_name}/ not found in {TEST_NEW_CROPS_DIR}')
            continue
        paths = sorted(p for p in cls_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        print(f'  {folder_name:<12} {len(paths):>3} images  (label {label_id})')
        samples.extend((p, label_id) for p in paths)
    return samples


def predict_test_new_crops(samples, scaler, mlp):
    """Returns y_true, y_pred, y_conf, results (list of dicts: path/gt/pred/conf/face_found)."""
    y_true, y_pred, y_conf, results = [], [], [], []
    total = len(samples)
    for i, (img_path, gt_label) in enumerate(samples, 1):
        print(f'\r  Processing {i}/{total} …', end='', flush=True)
        img = cv2.imread(str(img_path))
        feat = _extract_features_from_image(img) if img is not None else None
        if feat is None:
            results.append({'path': img_path, 'gt': gt_label,
                            'pred': None, 'conf': None, 'face_found': False})
            continue
        X_s  = scaler.transform(feat.reshape(1, -1))
        pred = int(mlp.predict(X_s)[0])
        conf = float(mlp.predict_proba(X_s)[0, pred])
        y_true.append(gt_label)
        y_pred.append(pred)
        y_conf.append(conf)
        results.append({'path': img_path, 'gt': gt_label,
                        'pred': pred, 'conf': conf, 'face_found': True})
    print()
    return y_true, y_pred, y_conf, results


def save_test_new_crops_report(results, y_true, y_pred, cm):
    lines = ['test_new_crops/ Predictions', '=' * 80,
             f'{"File":<55} {"GT":<12} {"Pred":<12} {"Conf":>6}  {"Result"}', '-' * 80]

    per_class = {i: {'total': 0, 'pass': 0, 'fail': 0, 'no_face': 0} for i in range(3)}
    for r in results:
        gt_name = LABEL_NAMES[r['gt']]
        per_class[r['gt']]['total'] += 1
        filename = r['path'].name
        if not r['face_found']:
            lines.append(f'{filename:<55} {gt_name:<12} {"—":<12} {"—":>6}  NO FACE')
            per_class[r['gt']]['no_face'] += 1
        else:
            pred_name = LABEL_NAMES[r['pred']]
            status    = 'PASS' if r['pred'] == r['gt'] else 'FAIL'
            lines.append(f'{filename:<55} {gt_name:<12} {pred_name:<12} {r["conf"]:>5.1%}  {status}')
            per_class[r['gt']]['pass' if r['pred'] == r['gt'] else 'fail'] += 1

    lines += ['', 'Per-class summary', '-' * 50]
    for lbl, name in LABEL_NAMES.items():
        d = per_class[lbl]
        scored  = d['pass'] + d['fail']
        acc_str = f"{d['pass']/scored:.1%}" if scored > 0 else 'n/a'
        lines.append(f'  {name:<12}  total={d["total"]}  pass={d["pass"]}  '
                     f'fail={d["fail"]}  no_face={d["no_face"]}  acc={acc_str}')

    if y_true:
        overall = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        lines += ['', f'OVERALL accuracy (faces only): {overall:.1%}  '
                      f'({sum(t == p for t, p in zip(y_true, y_pred))}/{len(y_true)})']

    lines += ['', 'Confusion matrix (rows=actual, cols=predicted):']
    header = f'{"":12}' + ''.join(f'{n:>12}' for n in LABEL_NAMES.values())
    lines.append(header)
    for r in range(3):
        lines.append(f'{LABEL_NAMES[r]:<12}' + ''.join(f'{cm[r, c]:>12d}' for c in range(3)))

    path = MODELS_DIR / 'test_new_crops_predictions.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  Saved test_new_crops_predictions.txt')


def compute_confidence_curves(y_test, proba):
    thresholds = np.linspace(0.01, 0.99, 200)
    data = {}
    for lbl, name in LABEL_NAMES.items():
        y_bin = (y_test == lbl).astype(int)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            pred = (proba[:, lbl] >= t).astype(int)
            tp = int(((pred == 1) & (y_bin == 1)).sum())
            fp = int(((pred == 1) & (y_bin == 0)).sum())
            fn = int(((pred == 0) & (y_bin == 1)).sum())
            p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        data[name] = {'precision': precisions, 'recall': recalls, 'f1': f1s}
    return thresholds, data


def compute_pr_curves(y_test, proba):
    pr_data = {}
    for lbl, name in LABEL_NAMES.items():
        y_bin = (y_test == lbl).astype(int)
        p, r, _ = precision_recall_curve(y_bin, proba[:, lbl])
        ap = average_precision_score(y_bin, proba[:, lbl])
        pr_data[name] = {'precision': p, 'recall': r, 'ap': ap}
    return pr_data


def plot_precision_recall_curve(pr_data):
    plt.figure(figsize=(7, 5))
    for lbl, name in LABEL_NAMES.items():
        d = pr_data[name]
        plt.plot(d['recall'], d['precision'],
                 label=f'{name} (AP={d["ap"]:.2f})', color=COLORS[lbl], linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save('precision_recall_curve.png')


def plot_confidence_curves(thresholds, conf_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, ylabel) in zip(axes, [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 Score')]):
        for lbl, name in LABEL_NAMES.items():
            ax.plot(thresholds, conf_data[name][key],
                    label=name, color=COLORS[lbl], linewidth=2)
        ax.set_title(f'{ylabel}-Confidence Curve')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('MLP Confidence Threshold Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('confidence_curves.png')


# ── Stats files ────────────────────────────────────────────────────────────────
def save_training_stats(mlp, X, X_train, X_test, y_test, y_pred, report):
    accuracy = accuracy_score(y_test, y_pred)
    path = MODELS_DIR / 'training_stats.txt'
    with open(path, 'w') as f:
        f.write('MLP Training Statistics\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Seed (all randomness)  : {SEED}\n')
        f.write(f'Input CSV              : dataset/labeled_features_dataset.csv\n\n')
        f.write('Dataset\n')
        f.write(f'  Total samples : {len(X)}\n')
        f.write(f'  Train samples : {len(X_train)}\n')
        f.write(f'  Test samples  : {len(X_test)}\n\n')
        f.write('Training\n')
        f.write(f'  Iterations    : {mlp.n_iter_}\n')
        f.write(f'  Final loss    : {mlp.loss_:.6f}\n\n')
        f.write('Evaluation\n')
        f.write(f'  Accuracy      : {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n')
        f.write('Classification Report\n')
        f.write('-' * 50 + '\n')
        f.write(report)
    print('  Saved training_stats.txt')


def save_curve_stats(mlp, thresholds, conf_data, pr_data):
    path = MODELS_DIR / 'curve_statistics.txt'
    col_w = 12
    with open(path, 'w') as f:
        f.write('MLP Curve Statistics\n')
        f.write('=' * 60 + '\n')

        _section(f, 'TRAINING LOSS CURVE (every 5 iterations)')
        loss = mlp.loss_curve_
        n = len(loss)
        indices = list(range(0, n, 5))
        if (n - 1) not in indices:
            indices.append(n - 1)
        f.write(f'{"Iteration":>10}  {"Loss":>10}\n')
        f.write('-' * 24 + '\n')
        for i in indices:
            f.write(f'{i + 1:>10}  {loss[i]:>10.6f}\n')

        stat_indices = [i for i, t in enumerate(thresholds)
                        if any(abs(t - st) < 0.006 for st in STAT_THRESHOLDS)]

        for metric, label in [('precision', 'PRECISION'), ('recall', 'RECALL'), ('f1', 'F1')]:
            _section(f, f'{label}-CONFIDENCE CURVE (threshold step 0.05)')
            header = f'{"Threshold":>10}' + ''.join(
                f'{n:>{col_w}}' for n in LABEL_NAMES.values())
            f.write(header + '\n')
            f.write('-' * (10 + col_w * len(LABEL_NAMES)) + '\n')
            for i in stat_indices:
                t = thresholds[i]
                row = f'{t:>10.2f}'
                for name in LABEL_NAMES.values():
                    row += f'{conf_data[name][metric][i]:>{col_w}.4f}'
                f.write(row + '\n')

        _section(f, 'PRECISION-RECALL CURVE (recall step ~0.05)')
        recall_steps = np.round(np.arange(0.0, 1.05, 0.05), 2)
        header = f'{"Recall":>8}' + ''.join(
            f'{n + " P":>{col_w}}' for n in LABEL_NAMES.values())
        f.write(header + '\n')
        f.write('-' * (8 + col_w * len(LABEL_NAMES)) + '\n')
        for rs in recall_steps:
            row = f'{rs:>8.2f}'
            for name in LABEL_NAMES.values():
                recalls_arr = pr_data[name]['recall']
                prec_arr    = pr_data[name]['precision']
                idx = np.argmin(np.abs(recalls_arr - rs))
                row += f'{prec_arr[idx]:>{col_w}.4f}'
            f.write(row + '\n')
        f.write('\nAverage Precision (AP) per class:\n')
        for name in LABEL_NAMES.values():
            f.write(f'  {name:<12}: {pr_data[name]["ap"]:.4f}\n')
    print('  Saved curve_statistics.txt')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        sys.exit(
            f'\ndataset/labeled_features_dataset.csv not found.\n'
            'Run extract_features_dataset.py first:\n'
            '  python src/classifiers/landmark_pipeline/extract_features_dataset.py\n'
        )

    df = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df)} rows from dataset/labeled_features_dataset.csv')
    print(df['label'].map(LABEL_NAMES).value_counts().to_string())

    plot_class_distribution(df)
    plot_feature_distributions(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label'].values.astype(int)

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f'\nTrain: {len(X_train)}  Test: {len(X_test)}  (seed={SEED}, stratified)')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train ──────────────────────────────────────────────────────────────────
    print('\n=== Training MLP ===')
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=SEED,
        verbose=False,
    )
    mlp.fit(X_train_s, y_train)
    print(f'Complete — {mlp.n_iter_} iterations, loss: {mlp.loss_:.4f}')
    plot_loss_curve(mlp)

    # ── Evaluate #1: held-out 20% dataset split ───────────────────────────────
    print('\n=== Evaluation: dataset 20% test split ===')
    y_pred = mlp.predict(X_test_s)
    proba  = mlp.predict_proba(X_test_s)
    report = classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()))
    print(report)

    plot_confusion_matrix(y_test, y_pred)
    thresholds, conf_data = compute_confidence_curves(y_test, proba)
    pr_data               = compute_pr_curves(y_test, proba)
    plot_precision_recall_curve(pr_data)
    plot_confidence_curves(thresholds, conf_data)
    save_training_stats(mlp, X, X_train, X_test, y_test, y_pred, report)
    save_curve_stats(mlp, thresholds, conf_data, pr_data)

    # ── Evaluate #2: test_new_crops/ (unseen real-world images) ──────────────
    print('\n=== Evaluation: test_new_crops/ ===')
    if not TEST_NEW_CROPS_DIR.exists():
        print(f'  [INFO] {TEST_NEW_CROPS_DIR} not found — skipping.')
    else:
        samples = collect_test_new_crops()
        if not samples:
            print(f'  No images found in {TEST_NEW_CROPS_DIR} — skipping.')
        else:
            print(f'\nRunning MLP on {len(samples)} images …')
            y_true_new, y_pred_new, _, results_new = predict_test_new_crops(
                samples, scaler, mlp)
            no_face = sum(1 for r in results_new if not r['face_found'])
            print(f'  {len(y_true_new)} predictions made  |  {no_face} skipped (no face)')

            if not y_true_new:
                print('  No faces detected in any test_new_crops image — skipping confusion matrix.')
            else:
                cm_new = plot_confusion_matrix(
                    y_true_new, y_pred_new,
                    tag='_test_new_crops', title_suffix=' — test_new_crops')
                save_test_new_crops_report(results_new, y_true_new, y_pred_new, cm_new)
                overall_new = sum(t == p for t, p in zip(y_true_new, y_pred_new)) / len(y_true_new)
                print(f'  test_new_crops accuracy: {overall_new:.1%}  '
                     f'({sum(t == p for t, p in zip(y_true_new, y_pred_new))}/{len(y_true_new)} faces)')

    # ── Save model ─────────────────────────────────────────────────────────────
    import joblib
    joblib.dump(mlp,    MODELS_DIR / 'mlp_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f'\nModel  → models/MLP_dataset/mlp_model.pkl')
    print(f'Scaler → models/MLP_dataset/scaler.pkl')


if __name__ == '__main__':
    main()
