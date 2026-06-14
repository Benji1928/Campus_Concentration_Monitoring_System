"""
Train MLP classifier — V1 Configuration.

V1: 3-layer [128, 64, 32], relu, Adam, alpha=0.001, early_stopping=True.

Usage (from project root):
    python src/classifiers/landmark_pipeline/train_evaluate_v1.py

Outputs (models/MLP_V1/):
    mlp_model.pkl, scaler.pkl
    training_stats.txt, curve_statistics.txt
    class_distribution.png, feature_distributions.png
    loss_curve.png, confusion_matrix.png, confusion_matrix_norm.png
    precision_recall_curve.png, confidence_curves.png
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
from src.classifiers.landmark_pipeline.mlp_configs import MLPV1Config

CONFIG = MLPV1Config()

DATA_PATH  = ROOT / 'data' / 'labeled_features.csv'
MODELS_DIR = ROOT / 'models' / CONFIG.output_dir
LABEL_NAMES = {0: 'ATTENTIVE', 1: 'DROWSY', 2: 'DISTRACTED'}
COLORS = {0: '#2ecc71', 1: '#e67e22', 2: '#e74c3c'}

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
    print(f'\nConfig: {CONFIG.description}')
    mlp = MLPClassifier(
        hidden_layer_sizes=CONFIG.hidden_layer_sizes,
        activation=CONFIG.activation,
        solver=CONFIG.solver,
        alpha=CONFIG.alpha,
        learning_rate_init=CONFIG.learning_rate_init,
        max_iter=CONFIG.max_iter,
        random_state=CONFIG.random_state,
        early_stopping=CONFIG.early_stopping,
        validation_fraction=CONFIG.validation_fraction,
        n_iter_no_change=CONFIG.n_iter_no_change,
        verbose=False,
    )
    mlp.fit(X_train_s, y_train)
    print(f'Training complete — {mlp.n_iter_} iterations, loss: {mlp.loss_:.4f}')
    return mlp


def plot_loss_curve(mlp):
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, color='steelblue', linewidth=2)
    if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
        ax2 = plt.gca().twinx()
        ax2.plot(mlp.validation_scores_, color='orange', linewidth=1.5,
                 linestyle='--', label='Val score')
        ax2.set_ylabel('Validation Score')
        ax2.legend(loc='lower right', fontsize=8)
    plt.title(f'MLP Training Loss Curve — {CONFIG.output_dir}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save('loss_curve.png')


# ── Curve computation ─────────────────────────────────────────────────────────
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
    plt.title(f'Confusion Matrix — {CONFIG.output_dir}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save('confusion_matrix.png')

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list, vmin=0, vmax=1)
    plt.title(f'Normalised Confusion Matrix — {CONFIG.output_dir}')
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
    plt.title(f'Precision-Recall Curve — {CONFIG.output_dir}')
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
    plt.suptitle(f'MLP Confidence Threshold Curves — {CONFIG.output_dir}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('confidence_curves.png')


# ── Statistics files ──────────────────────────────────────────────────────────
def save_training_stats(mlp, X, X_train, X_test, y_test, y_pred, report):
    accuracy = accuracy_score(y_test, y_pred)
    path = MODELS_DIR / 'training_stats.txt'
    with open(path, 'w') as f:
        f.write(f'MLP Training Statistics — {CONFIG.output_dir}\n')
        f.write('=' * 50 + '\n\n')
        f.write('Configuration\n')
        f.write(f'  {CONFIG.description}\n\n')
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


def save_curve_stats(mlp, thresholds, conf_data, pr_data):
    path = MODELS_DIR / 'curve_statistics.txt'
    col_w = 12

    with open(path, 'w') as f:
        f.write(f'MLP Curve Statistics — {CONFIG.output_dir} — sampled at intervals of 5\n')
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

    print(f'Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {MODELS_DIR}')
    print(f'Config: {CONFIG.description}\n')

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
    save_curve_stats(mlp, thresholds, conf_data, pr_data)

    import joblib
    joblib.dump(mlp,    MODELS_DIR / 'mlp_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f'\nModel  -> {MODELS_DIR}/mlp_model.pkl')
    print(f'Scaler -> {MODELS_DIR}/scaler.pkl')


if __name__ == '__main__':
    main()
