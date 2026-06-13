"""
CNN Training — V2 Configuration.

V2: MobileNetV3-Large, deeper head [512, 256], heavy dropout=0.5,
    label smoothing=0.1, extra augmentation (perspective warp + gaussian blur
    + random erasing), LR=5e-5, BS=32, patience=7.

Usage:
    python src/classifiers/classification_pipeline/CNN_training_v2.py

If running locally, set CNNV2Config.dataset_dir to your dataset path.
Outputs are saved to models/CNN_V2/ (local) or /kaggle/working/CNN_V2/ (Kaggle).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import timm
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent.parent

import sys
sys.path.insert(0, str(ROOT))

from src.classifiers.classification_pipeline.cnn_configs import CNNV2Config


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = CNNV2Config()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_paths():
    ds = Path(CONFIG.dataset_dir)
    if not ds.exists():
        local = ROOT / 'data' / 'designproject-clean'
        if local.exists():
            ds = local
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{CONFIG.dataset_dir}'.\n"
                "Set CNNV2Config.dataset_dir to your local dataset path."
            )

    kaggle_working = Path('/kaggle/working')
    if kaggle_working.exists():
        out = kaggle_working / Path(CONFIG.output_dir).name
    else:
        out = ROOT / CONFIG.output_dir

    out.mkdir(parents=True, exist_ok=True)
    return ds, out


# ── Activation helper ─────────────────────────────────────────────────────────
def get_activation(name):
    return nn.Tanh() if name == 'tanh' else nn.ReLU()


# ── Classifier head builder ───────────────────────────────────────────────────
def build_head(num_features, hidden_sizes, num_classes, dropout, activation):
    layers = []
    in_size = num_features
    for h in hidden_sizes:
        layers.extend([
            nn.Linear(in_size, h),
            nn.BatchNorm1d(h),
            get_activation(activation),
            nn.Dropout(dropout),
        ])
        in_size = h
    layers.append(nn.Linear(in_size, num_classes))
    return nn.Sequential(*layers)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dataset_dir, output_dir = _resolve_paths()
    print(f'Config : {CONFIG.description}')
    print(f'Dataset: {dataset_dir}')
    print(f'Output : {output_dir}')
    print(f'Device : {DEVICE}\n')

    # ── Transforms ────────────────────────────────────────────────────────────
    # Extra augmentation includes perspective warp, gaussian blur, and erasing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Stratified split ──────────────────────────────────────────────────────
    train_full = datasets.ImageFolder(root=str(dataset_dir), transform=train_transform)
    val_full   = datasets.ImageFolder(root=str(dataset_dir), transform=val_transform)
    class_names = train_full.classes

    indices = np.arange(len(train_full))
    targets = np.array(train_full.targets)
    train_idx, val_idx = train_test_split(
        indices, test_size=CONFIG.test_size, stratify=targets, random_state=42
    )

    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=CONFIG.batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx),
        batch_size=CONFIG.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f'Train: {len(train_idx)}  Val: {len(val_idx)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'Loading {CONFIG.model_name} backbone...')
    model = timm.create_model(CONFIG.model_name, pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = build_head(
        num_features, CONFIG.head_hidden, CONFIG.num_classes, CONFIG.dropout, CONFIG.activation
    )
    model = model.to(DEVICE)

    # Label smoothing penalises overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(CONFIG.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{CONFIG.epochs}'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_idx)
        epoch_train_acc  = correct_train / total_train * 100

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_idx)
        epoch_val_acc  = correct_val / total_val * 100

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f'   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%  '
              f'|  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%')

        if epoch_val_loss < (best_val_loss - CONFIG.min_delta):
            best_val_loss = epoch_val_loss
            best_val_acc  = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), str(output_dir / 'best_model.pth'))
            print('   [Saved] Val loss improved')
        else:
            patience_counter += 1
            print(f'   [Patience {patience_counter}/{CONFIG.patience}] No val improvement')

        if patience_counter >= CONFIG.patience:
            print(f'\nEarly stopping at epoch {epoch + 1}.')
            break

    print(f'\nTraining complete. Best val accuracy: {best_val_acc:.2f}%')

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(str(output_dir / 'best_model.pth'), map_location=DEVICE))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)

    report = classification_report(all_targets, all_preds, target_names=class_names)
    print('\nClassification Report\n' + '=' * 50)
    print(report)

    # ── Training curves ───────────────────────────────────────────────────────
    epochs_ran = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, epochs_ran + 1), train_losses, label='Train', color='steelblue')
    axes[0].plot(range(1, epochs_ran + 1), val_losses,   label='Val',   color='orange')
    axes[0].set_title(f'Loss — {CONFIG.output_dir}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, epochs_ran + 1), train_accs, label='Train', color='steelblue')
    axes[1].plot(range(1, epochs_ran + 1), val_accs,   label='Val',   color='orange')
    axes[1].set_title(f'Accuracy — {CONFIG.output_dir}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'training_curves.png'), dpi=150)
    plt.close()
    print(f'Saved {output_dir}/training_curves.png')

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
    plt.title(f'Confusion Matrix — {CONFIG.output_dir}', fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'Saved {output_dir}/confusion_matrix.png')

    # ── Stats file ────────────────────────────────────────────────────────────
    accuracy = (all_preds == all_targets).mean()
    stats_path = output_dir / 'training_stats.txt'
    with open(stats_path, 'w') as f:
        f.write(f'CNN Training Statistics — {CONFIG.output_dir}\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Config: {CONFIG.description}\n\n')
        f.write(f'Dataset     : {dataset_dir}\n')
        f.write(f'Train split : {len(train_idx)}\n')
        f.write(f'Val split   : {len(val_idx)}\n\n')
        f.write(f'Epochs run  : {epochs_ran}\n')
        f.write(f'Best val loss    : {best_val_loss:.6f}\n')
        f.write(f'Best val accuracy: {best_val_acc:.2f}%\n')
        f.write(f'Final accuracy   : {accuracy * 100:.2f}%\n\n')
        f.write('Classification Report\n')
        f.write('-' * 60 + '\n')
        f.write(report)
        f.write('\nPer-epoch summary:\n')
        f.write(f'  {"Epoch":>5}  {"TrainLoss":>10}  {"ValLoss":>10}  '
                f'{"TrainAcc":>9}  {"ValAcc":>8}\n')
        f.write('  ' + '-' * 50 + '\n')
        for i, (tl, vl, ta, va) in enumerate(
                zip(train_losses, val_losses, train_accs, val_accs), 1):
            f.write(f'  {i:>5}  {tl:>10.6f}  {vl:>10.6f}  '
                    f'{ta:>9.2f}  {va:>8.2f}\n')
    print(f'Saved {stats_path}')
    print(f'\nDone — all outputs in {output_dir}')


if __name__ == '__main__':
    main()
