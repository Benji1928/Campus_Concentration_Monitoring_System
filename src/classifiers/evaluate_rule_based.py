import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.classifiers.landmark_pipeline.rule_based import RuleBasedClassifier
from src.classifiers.landmark_pipeline.face_mesh import FaceMesh
from src.classifiers.landmark_pipeline.feature_extractor import FeatureExtractor


TEST_DIR = ROOT / "test_new_crops"

LABEL_MAP = {
    "ATTENTIVE": 0,
    "DROWSY": 1,
    "DISTRACTED": 2,
}

OUT_DIR = ROOT / "models" / "RuleBased"
COLORS = {0: '#2ecc71', 1: '#e67e22', 2: '#e74c3c'}


def _save(filename):
    path = OUT_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_confusion_matrix(y_true, y_pred):
    label_list = list(LABEL_MAP.keys())
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list)
    plt.title('Confusion Matrix — Rule-Based')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save('confusion_matrix.png')

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list, vmin=0, vmax=1)
    plt.title('Normalised Confusion Matrix — Rule-Based')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save('confusion_matrix_norm.png')


def _pick_diverse(subset_df, n=3):
    """Pick up to n rows, preferring one from each class for variety."""
    picked = []
    for class_name in LABEL_MAP:
        cls_rows = subset_df[subset_df["ground_truth"] == class_name]
        if not cls_rows.empty:
            picked.append(cls_rows.iloc[0])
        if len(picked) == n:
            break
    # Fill remaining slots if we didn't get enough variety
    remaining = subset_df[~subset_df.index.isin([r.name for r in picked])]
    for _, row in remaining.iterrows():
        if len(picked) == n:
            break
        picked.append(row)
    return picked


def _tile(image_path, label_top, label_bot, border_color, size=224):
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    border = 4
    img[:border, :] = border_color
    img[-border:, :] = border_color
    img[:, :border] = border_color
    img[:, -border:] = border_color

    return img, label_top, label_bot


def plot_best_worst(df):
    correct_df   = df[df["correct"] == True]
    incorrect_df = df[df["correct"] == False]

    best  = _pick_diverse(correct_df,   n=3)
    worst = _pick_diverse(incorrect_df, n=3)

    GREEN = (60, 179, 60)
    RED   = (220, 50, 50)

    tiles = []
    for row in best:
        tiles.append(_tile(row["path"], f"GT: {row['ground_truth']}", f"Pred: {row['prediction']}", GREEN))
    for row in worst:
        tiles.append(_tile(row["path"], f"GT: {row['ground_truth']}", f"Pred: {row['prediction']}", RED))

    _, axes = plt.subplots(2, 3, figsize=(10, 7))
    row_labels = ["Best (Correct)", "Worst (Incorrect)"]

    for ax, (img, top, bot) in zip(axes.flatten(), tiles):
        ax.imshow(img)
        ax.set_title(f"{top}\n{bot}", fontsize=9)
        ax.axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold", labelpad=10)

    plt.suptitle("Best 3 vs Worst 3 Predictions — Rule-Based", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("best_worst.png")


def evaluate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier = RuleBasedClassifier()
    face_mesh = FaceMesh()

    y_true = []
    y_pred = []
    rows = []
    skipped = 0

    for class_name, gt_id in LABEL_MAP.items():
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            print(f"[WARN] Directory not found: {class_dir}")
            continue

        for image_path in class_dir.glob("*"):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Failed to read: {image_path.name}")
                skipped += 1
                continue

            frame_h, frame_w = frame.shape[:2]
            extractor = FeatureExtractor(frame_w, frame_h)
            landmarks = face_mesh.process(frame)

            if landmarks is None:
                print(f"No face detected: {image_path.name}")
                skipped += 1
                continue

            try:
                features = extractor.extract(landmarks)
                pred_name, pred_id = classifier.predict(features)

                y_true.append(gt_id)
                y_pred.append(pred_id)

                rows.append({
                    "image": image_path.name,
                    "path": str(image_path),
                    "ground_truth": class_name,
                    "prediction": pred_name,
                    "correct": pred_id == gt_id,
                })

            except Exception as e:
                print(f"Failed: {image_path.name} — {e}")
                skipped += 1

    face_mesh.close()

    if not y_true:
        print("No samples evaluated. Check TEST_DIR structure.")
        return None

    df = pd.DataFrame(rows)
    results_path = OUT_DIR / "rule_based_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Saved {results_path}")

    report: str = classification_report(y_true, y_pred, target_names=list(LABEL_MAP.keys()), output_dict=False)  # type: ignore[assignment]
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nEvaluated {len(y_true)} images  ({skipped} skipped)")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report")
    print("=" * 50)
    print(report)

    plot_confusion_matrix(y_true, y_pred)
    plot_best_worst(df)

    stats_path = OUT_DIR / "eval_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Rule-Based Classifier Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total evaluated : {len(y_true)}\n")
        f.write(f"Skipped         : {skipped}\n\n")
        f.write(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    print(f"Saved {stats_path}")

    return df


if __name__ == "__main__":
    evaluate()
