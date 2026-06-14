"""
Train MLP classifier and generate evaluation graphs.

Usage (from project root):
    python src/classifiers/landmark_pipeline/train_evaluate.py

Outputs (models/MLP/):
    mlp_model.pkl, scaler.pkl
    training_stats.txt
    curve_statistics.txt
    class_distribution.png
    feature_distributions.png
    loss_curve.png
    confusion_matrix.png
    confusion_matrix_norm.png
    precision_recall_curve.png
    confidence_curves.png
    yolo_loss_curves.png       (only if models/results.csv exists)
    yolo_metric_curves.png     (only if models/results.csv exists)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_curve, average_precision_score,
)

from src.classifiers.landmark_pipeline.feature_extractor import FEATURE_COLS

DATA_PATH  = ROOT / 'data' / 'labeled_features.csv'
MODELS_DIR = ROOT / 'models' / 'MLP'
LABEL_NAMES = {0: 'ATTENTIVE', 1: 'DROWSY', 2: 'DISTRACTED'}
COLORS = {0: '#2ecc71', 1: '#e67e22', 2: '#e74c3c'}

# Confidence thresholds sampled at 0.05 intervals for stats file
STAT_THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(filename):
    path = MODELS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved {path}')


def _section(f, title, char='='):
    f.write(f'\n{title}\n{char * len(title)}\n')


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    if not DATA_PATH.exists():
        print(f'No data at {DATA_PATH}. Run collect_dataset.py first.')
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f'Loaded {len(df)} samples')
    print(df['label'].map(LABEL_NAMES).value_counts().to_string())
    return df


# ── EDA plots ─────────────────────────────────────────────────────────────────
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


# ── Training ──────────────────────────────────────────────────────────────────
def train_mlp(X_train_s, y_train):
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_s, y_train)
    print(f'Training complete — {mlp.n_iter_} iterations, loss: {mlp.loss_:.4f}')
    return mlp


def plot_loss_curve(mlp):
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, color='steelblue', linewidth=2)
    plt.title('MLP Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save('loss_curve.png')


# ── Curve computation (shared between plots and stats) ────────────────────────
def compute_confidence_curves(y_test, proba):
    """Returns per-class precision, recall, F1 at each threshold in STAT_THRESHOLDS."""
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
    """Returns per-class precision-recall curve data and average precision."""
    pr_data = {}
    for lbl, name in LABEL_NAMES.items():
        y_bin = (y_test == lbl).astype(int)
        p, r, thresholds = precision_recall_curve(y_bin, proba[:, lbl])
        ap = average_precision_score(y_bin, proba[:, lbl])
        pr_data[name] = {'precision': p, 'recall': r, 'thresholds': thresholds, 'ap': ap}
    return pr_data


# ── Evaluation plots ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    label_list = list(LABEL_NAMES.values())
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save('confusion_matrix.png')

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list, vmin=0, vmax=1)
    plt.title('Normalised Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save('confusion_matrix_norm.png')


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
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 Score')]
    for ax, (key, ylabel) in zip(axes, metrics):
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


# ── YOLO plots ────────────────────────────────────────────────────────────────
def _smooth(values, weight=0.6):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot_yolo_curves(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    epochs = range(1, len(df) + 1)

    loss_cols = {
        'train/box_loss': 'train/box_loss', 'train/cls_loss': 'train/cls_loss',
        'train/dfl_loss': 'train/dfl_loss', 'val/box_loss':   'val/box_loss',
        'val/cls_loss':   'val/cls_loss',   'val/dfl_loss':   'val/dfl_loss',
    }
    available_loss = {k: v for k, v in loss_cols.items() if k in df.columns}
    if available_loss:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, col in zip(axes.flatten(), available_loss):
            vals = df[col].tolist()
            ax.plot(epochs, vals, 'o-', color='steelblue', markersize=3, linewidth=1.5, label='results')
            ax.plot(epochs, _smooth(vals), '--', color='orange', linewidth=1.5, label='smooth')
            ax.set_title(col)
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)
        plt.suptitle('YOLO Training / Validation Loss', fontsize=13, fontweight='bold')
        plt.tight_layout()
        _save('yolo_loss_curves.png')

    metric_cols = {
        'metrics/precision(B)': 'metrics/precision(B)',
        'metrics/recall(B)':    'metrics/recall(B)',
        'metrics/mAP50(B)':     'metrics/mAP50(B)',
        'metrics/mAP50-95(B)':  'metrics/mAP50-95(B)',
    }
    available_metrics = {k: v for k, v in metric_cols.items() if k in df.columns}
    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
        if len(available_metrics) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_metrics):
            vals = df[col].tolist()
            ax.plot(epochs, vals, 'o-', color='steelblue', markersize=3, linewidth=1.5, label='results')
            ax.plot(epochs, _smooth(vals), '--', color='orange', linewidth=1.5, label='smooth')
            ax.set_title(col)
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)
        plt.suptitle('YOLO Evaluation Metrics', fontsize=13, fontweight='bold')
        plt.tight_layout()
        _save('yolo_metric_curves.png')

    return df, list(available_loss.keys()), list(available_metrics.keys())


# ── Statistics files ──────────────────────────────────────────────────────────
def save_training_stats(mlp, X, X_train, X_test, y_test, y_pred, report):
    accuracy = accuracy_score(y_test, y_pred)
    path = MODELS_DIR / 'training_stats.txt'
    with open(path, 'w') as f:
        f.write('MLP Training Statistics\n')
        f.write('=' * 50 + '\n\n')
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
    print(f'Saved {path}')


def save_curve_stats(mlp, thresholds, conf_data, pr_data, yolo_df=None,
                     yolo_loss_cols=None, yolo_metric_cols=None):
    """Writes curve_statistics.txt with values sampled every 5 steps."""
    path = MODELS_DIR / 'curve_statistics.txt'
    col_w = 12  # column width

    with open(path, 'w') as f:
        f.write('MLP Curve Statistics — sampled at intervals of 5\n')
        f.write('=' * 60 + '\n')

        # ── Training loss (every 5 iterations) ───────────────────────
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

        # ── Confidence curves (threshold step 0.05) ───────────────────
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

        # ── Precision-Recall curve (sampled at recall 0.05 steps) ────
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
        # Average precision summary
        f.write('\nAverage Precision (AP) per class:\n')
        for name in LABEL_NAMES.values():
            f.write(f'  {name:<12}: {pr_data[name]["ap"]:.4f}\n')

        # ── YOLO curves (every 5 epochs) ──────────────────────────────
        if yolo_df is not None and not yolo_df.empty:
            _section(f, 'YOLO TRAINING CURVES (every 5 epochs)')
            n_epochs = len(yolo_df)
            epoch_indices = list(range(0, n_epochs, 5))
            if (n_epochs - 1) not in epoch_indices:
                epoch_indices.append(n_epochs - 1)

            all_cols = (yolo_loss_cols or []) + (yolo_metric_cols or [])
            if all_cols:
                header = f'{"Epoch":>6}' + ''.join(f'{c[:14]:>16}' for c in all_cols)
                f.write(header + '\n')
                f.write('-' * (6 + 16 * len(all_cols)) + '\n')
                for i in epoch_indices:
                    row = f'{i + 1:>6}'
                    for col in all_cols:
                        val = yolo_df.iloc[i].get(col, float('nan'))
                        row += f'{val:>16.6f}'
                    f.write(row + '\n')

    print(f'Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    plot_class_distribution(df)
    plot_feature_distributions(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f'\nTrain: {len(X_train)}  Test: {len(X_test)}')

    mlp    = train_mlp(X_train_s, y_train)
    plot_loss_curve(mlp)

    y_pred = mlp.predict(X_test_s)
    proba  = mlp.predict_proba(X_test_s)
    report = classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()))
    print('\nClassification Report\n' + '=' * 50)
    print(report)

    plot_confusion_matrix(y_test, y_pred)

    thresholds, conf_data = compute_confidence_curves(y_test, proba)
    pr_data               = compute_pr_curves(y_test, proba)
    plot_precision_recall_curve(pr_data)
    plot_confidence_curves(thresholds, conf_data)

    save_training_stats(mlp, X, X_train, X_test, y_test, y_pred, report)

    # YOLO curves
    yolo_df, yolo_loss_cols, yolo_metric_cols = None, [], []
    yolo_results = ROOT / 'models' / 'results.csv'
    if yolo_results.exists():
        print('\n=== YOLO results.csv found ===')
        yolo_df, yolo_loss_cols, yolo_metric_cols = plot_yolo_curves(yolo_results)
    else:
        print(f'\n[INFO] No YOLO results.csv found — skipping YOLO curves.')
        print('       Copy your YOLO results.csv into models/ to generate them.')

    save_curve_stats(mlp, thresholds, conf_data, pr_data,
                     yolo_df, yolo_loss_cols, yolo_metric_cols)

    import joblib
    joblib.dump(mlp,    MODELS_DIR / 'mlp_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f'\nModel  -> {MODELS_DIR}/mlp_model.pkl')
    print(f'Scaler -> {MODELS_DIR}/scaler.pkl')


if __name__ == '__main__':
    main()
